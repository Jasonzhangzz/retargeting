"""Small synthetic smoke tests; no production training or learned-quality claim."""
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from retarget.env.grasp.manifold import (
    DenoisingModel,
    ExperimentData,
    Predictor,
    Subspace,
    TrainConfig,
    write_json,
)
from retarget.env.grasp.manifold_flow import (
    FlowConfig,
    PoseFlowModel,
    draw_pairs,
    finger_corruption,
    load_flow_model,
    pair_table,
    supervised_contrastive,
    train_flow,
)


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(6)
    size = np.tile(np.linspace(.6, 1.4, 7), 2)
    group = np.repeat(np.arange(2), 7)
    split = np.tile([0, 0, 1, 0, 2, 0, 0], 2)
    x = rng.normal(0, .1, (14, 20)) + size[:, None] * np.linspace(-.2, .2, 20) + group[:, None] * .1
    qpos = np.c_[np.zeros((14, 6)), x]
    np.savez_compressed(tmp_path / "dataset.npz", qpos=qpos, group=group, size=size, split=split,
                        source_index=np.arange(14), mid=np.zeros(20), half=np.ones(20))
    metadata = dict(version=1, source="missing-smoke-source", source_sha256={},
                    groups=[dict(template="1_Type", primitive="cylinder"), dict(template="2_Type", primitive="sphere")],
                    subspaces=[Subspace.fit(x[(group == g) & (split == 0)]).record() for g in range(2)],
                    log_size_center=0., log_size_scale=1.)
    write_json(tmp_path / "dataset.json", metadata)
    return ExperimentData(tmp_path, check_source=False)


def small_config(**kwargs):
    return replace(FlowConfig(steps=2, batch_size=8, hidden_dim=32, latent_dim=4,
                             codebook_size=8, flow_steps=4, min_steps=0, validation_every=1), **kwargs)


def test_finger_noise_zero_five_independent_rounds_and_reproducibility():
    clean = torch.zeros(3, 20)
    counts = torch.tensor([0, 5, 2])
    rounds = torch.tensor([[1, 2, 3, 4, 5]] * 3)
    def sample():
        return finger_corruption(clean, torch.Generator().manual_seed(11), sigma_min=.05, sigma_max=.05,
                                 finger_counts=counts, rounds=rounds)
    observed, info = sample()
    torch.testing.assert_close(observed, sample()[0], rtol=0, atol=0)
    torch.testing.assert_close(observed, info["increments"].sum(0), rtol=0, atol=0)
    assert not observed[0].any()
    torch.testing.assert_close(info["mask"].sum(-1), counts)
    assert set(info["rounds"][1].tolist()) == {1, 2, 3, 4, 5}
    assert not observed.reshape(3, 5, 4)[~info["mask"]].any()
    for r in range(5):
        inactive = info["rounds"] <= r
        assert not info["increments"][r].reshape(3, 5, 4)[inactive].any()
    n = 4000
    samples, _ = finger_corruption(torch.zeros(n, 20), torch.Generator().manual_seed(12),
                                   sigma_min=.05, sigma_max=.05, finger_counts=torch.full((n,), 5),
                                   rounds=torch.full((n, 5), 5))
    assert float(samples.var()) == pytest.approx(5 * .05 ** 2, rel=.04)


@pytest.mark.parametrize("kwargs", [dict(max_rounds=0), dict(sigma_min=0), dict(sigma_max=float("nan")),
                                  dict(finger_counts=[6]), dict(rounds=[[0] * 5])])
def test_corruption_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        finger_corruption(torch.zeros(1, 20), torch.Generator(), **kwargs)


@pytest.mark.parametrize("kind", ["vq", "vae"])
def test_autoencoder_objective_gradients_and_frozen_flow(kind, synthetic_data):
    torch.manual_seed(4)
    torch.set_num_threads(2)
    config = small_config()
    model = PoseFlowModel(kind, synthetic_data, config)
    x = torch.tensor(synthetic_data.x, dtype=torch.float32)
    g = torch.tensor(synthetic_data.group)
    s = torch.tensor(synthetic_data.size, dtype=torch.float32)
    table, counts, unique = pair_table(g, s)
    ids, partners = draw_pairs(table, counts, unique, torch.tensor([0, 1, 0, 1]), torch.Generator().manual_seed(1))
    assert torch.equal(g[ids], g[ids][partners])
    assert (s[ids] != s[ids][partners]).all()
    loss, components = model.autoencoder_objective(x[ids], g[ids], s[ids], partners, torch.Generator().manual_seed(3))
    assert {"cross_size", "consistency", "contrastive", "reconstruction"} <= components.keys()
    optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=.01)
    before = {k: v.clone() for k, v in model.autoencoder.state_dict().items()}
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.autoencoder.parameters() if p.grad is not None)
    optimizer.step()
    if kind == "vq":
        assert not torch.equal(before["codebook.weight"], model.autoencoder.codebook.weight)
    else:
        mean, _ = model.autoencoder.encode(x)
        sample, _ = model.autoencoder.encode(x, sample=True, generator=torch.Generator().manual_seed(1))
        assert not torch.equal(mean, sample)
    model.freeze_autoencoder()
    model.train()
    assert not model.autoencoder.training
    frozen = {k: v.clone() for k, v in model.autoencoder.state_dict().items()}
    velocity = {k: v.clone() for k, v in model.velocity.state_dict().items()}
    optimizer = torch.optim.AdamW(model.velocity.parameters(), lr=.01)
    model.flow_objective(x, g, s, torch.Generator().manual_seed(5)).backward()
    assert all(p.grad is None for p in model.autoencoder.parameters())
    optimizer.step()
    assert all(torch.equal(v, model.autoencoder.state_dict()[k]) for k, v in frozen.items())
    assert any(not torch.equal(v, model.velocity.state_dict()[k]) for k, v in velocity.items())


def test_supervised_contrastive_prefers_same_type_clustering():
    features = torch.tensor([[1., 0.], [1., .01], [0., 1.], [.01, 1.]])
    assert supervised_contrastive(features, torch.tensor([0, 0, 1, 1])) < supervised_contrastive(
        features, torch.tensor([0, 1, 0, 1]))


class ConstantVelocity(nn.Module):
    def forward(self, x, times, latent, types, primitives, sizes):
        return torch.ones_like(x) * .1


class TimeVelocity(nn.Module):
    def forward(self, x, times, latent, types, primitives, sizes):
        return times[:, None] * x


def test_heun_starts_at_observation_has_correct_direction_and_converges(synthetic_data):
    model = PoseFlowModel("vae", synthetic_data, small_config())
    x, g, s = torch.ones(2, 20), torch.tensor([0, 1]), torch.ones(2)
    model.velocity = ConstantVelocity()
    original_encode, encoder_calls = model.autoencoder.encode, []
    def counted_encode(observed):
        encoder_calls.append(observed.clone())
        return original_encode(observed)
    model.autoencoder.encode = counted_encode
    torch.testing.assert_close(model.denoise(x, g, s), x + .1)
    assert len(encoder_calls) == 1
    torch.testing.assert_close(encoder_calls[0], x)
    # No random sampling and no PCA postprojection, including clean inputs.
    torch.testing.assert_close(model.denoise(x, g, s), model.denoise(x, g, s), rtol=0, atol=0)
    model.velocity = TimeVelocity()
    truth = torch.full_like(x, np.exp(.5))
    coarse = (model.denoise(x, g, s, steps=4) - truth).abs().max()
    fine = (model.denoise(x, g, s, steps=32) - truth).abs().max()
    assert fine < coarse / 10
    with pytest.raises(ValueError):
        model.denoise(x, g, s, steps=0)


@pytest.mark.parametrize("kind", ["vq", "vae"])
def test_two_stage_save_load_and_predict_smoke(kind, synthetic_data):
    config = small_config()
    result = train_flow(synthetic_data, kind, seed=7, config=config, device="cpu")
    assert result["stages"]["flow"]["frozen_autoencoder_unchanged"]
    method = f"diffusion_{kind}"
    ae, _ = load_flow_model(synthetic_data, method, 7, "cpu", stage="autoencoder")
    flow, checkpoint = load_flow_model(synthetic_data, method, 7, "cpu")
    assert checkpoint["autoencoder_sha256"]
    for key, value in ae.autoencoder.state_dict().items():
        torch.testing.assert_close(value, flow.autoencoder.state_dict()[key], rtol=0, atol=0)
    predictor = Predictor(synthetic_data, method, device="cpu")
    x, groups, sizes = synthetic_data.x[:3], synthetic_data.group[:3], synthetic_data.size[:3]
    a = predictor.predict(x, groups, sizes)
    b = Predictor(synthetic_data, method, device="cpu").predict(x, groups, sizes, sigma=.3)
    assert a.shape == (3, 20) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    # Sigma is not a flow condition and clean inputs still run the network.
    predictor.model.velocity = ConstantVelocity()
    np.testing.assert_allclose(predictor.predict(x, groups, sizes, 0), x + .1, atol=2e-7)
    with pytest.raises(FileExistsError):
        train_flow(synthetic_data, kind, 7, config, "cpu")
    with pytest.raises(ValueError, match="requires"):
        train_flow(synthetic_data, kind, 8, config, "cpu", stage="flow")
    source = synthetic_data.directory / "models" / f"{method}_seed7" / "autoencoder.pt"
    independent = train_flow(synthetic_data, kind, 8, config, "cpu", stage="flow", autoencoder_path=source)
    assert independent["stages"]["flow"]["frozen_autoencoder_unchanged"]
    with pytest.raises(ValueError, match="Incompatible"):
        load_flow_model(synthetic_data, method, 7, "cpu", stage="flow", path=source)
    wrong_kind = "diffusion_vae" if kind == "vq" else "diffusion_vq"
    with pytest.raises(ValueError, match="Incompatible"):
        load_flow_model(synthetic_data, wrong_kind, 7, "cpu", stage="autoencoder", path=source)
    with pytest.raises(ValueError, match="incompatible"):
        train_flow(synthetic_data, kind, 9, replace(config, latent_dim=6), "cpu", stage="flow", autoencoder_path=source)
    synthetic_data.directory.joinpath("dataset.json").write_text(
        synthetic_data.directory.joinpath("dataset.json").read_text() + "\n")
    with pytest.raises(ValueError, match="dataset"):
        load_flow_model(synthetic_data, method, 7, "cpu")


def test_explicit_new_model_does_not_fall_back_to_legacy(synthetic_data):
    with pytest.raises(FileNotFoundError):
        Predictor(synthetic_data, "diffusion_vq", device="cpu")
    write_json(synthetic_data.directory / "models.json", dict(default_diffusion="diffusion_vq"))
    with pytest.raises(FileNotFoundError):
        Predictor(synthetic_data, "diffusion", device="cpu")


def test_flow_objective_matches_observation_to_clean_velocity(synthetic_data):
    model = PoseFlowModel("vq", synthetic_data, small_config())
    model.freeze_autoencoder()
    x = torch.tensor(synthetic_data.x[:4], dtype=torch.float32)
    groups, sizes = torch.zeros(4, dtype=torch.long), torch.ones(4)
    y, _ = model.corrupt(x, torch.Generator().manual_seed(32))
    class Oracle(nn.Module):
        def forward(self, current, t, z, types, primitives, s):
            torch.testing.assert_close(current, torch.lerp(y, x, t[:, None]))
            return x - y
    model.velocity = Oracle()
    assert model.flow_objective(x, groups, sizes, torch.Generator().manual_seed(32)) == 0


def test_partial_finger_inspector_export_smoke(synthetic_data, monkeypatch):
    from retarget.env.grasp import denoising_inspector as inspector
    train_flow(synthetic_data, "vae", 7, small_config(), "cpu")
    primitive = dict(size_axis="radius", dimensions_m=[.05, .05, .1])
    synthetic_data.snapshot = SimpleNamespace(manifest=dict(cases=[dict(primitive=primitive)] * 14))
    monkeypatch.setattr(inspector, "PoseChecker", lambda data, index: SimpleNamespace(model=None))
    monkeypatch.setattr(inspector, "visual_model", lambda model, meshes, group: {})
    monkeypatch.setattr(inspector, "quick_metrics", lambda checker, q, reference, half:
                        dict(rmse=float(np.sqrt(np.mean((q-reference)**2))), enclosure_mm=0.,
                             object_mm=0., self_mm=0., violation_rad=0.))
    output = synthetic_data.directory / "inspector"
    html = inspector.build_inspector(synthetic_data, output, rounds=2, seeds=(11,),
                                     fractions=(0.,), method="diffusion_vae", device="cpu")
    payload = json.loads((output / "experiment.json").read_text())
    assert payload["flow"] and payload["diffusion_method"] == "diffusion_vae"
    assert "const DATA=" in html.read_text()
    for case in payload["cases"]:
        path = case["paths"][0]
        assert len(path["finger_rounds"]) == 5
        for record in path["rounds"]:
            assert "timestep" not in record
            assert (record["integration_start"], record["integration_end"]) == (0., 1.)
            np.testing.assert_array_equal(np.array(record["observed"])[:6], case["truth"][:6])
    verification = json.loads((output / "verification.json").read_text())
    assert verification["gaussian_addback_max_error"] < 1e-14
    with np.load(output / "poses.npz") as poses:
        assert poses["qpos"].shape == (12, 5, 26)


def test_comparison_evaluation_smoke(synthetic_data, monkeypatch):
    """Use tiny random legacy weights; this checks plumbing, not quality."""
    import hashlib
    from dataclasses import asdict

    from retarget.env.grasp import manifold_flow_evaluation as evaluation
    for kind in ("vq", "vae"):
        train_flow(synthetic_data, kind, 7, small_config(), "cpu")
    baseline_directory = synthetic_data.directory / "legacy"
    baseline_directory.mkdir()
    for name in ("dataset.json", "dataset.npz"):
        shutil.copy2(synthetic_data.directory / name, baseline_directory / name)
    baseline = ExperimentData(baseline_directory, check_source=False)
    config = TrainConfig(diffusion_steps=10, ddim_steps=2)
    model = DenoisingModel("diffusion", baseline, config)
    directory = baseline_directory / "models" / "diffusion_seed7"
    directory.mkdir(parents=True)
    torch.save(dict(state=model.state_dict(), config=asdict(config), method="diffusion", seed=7,
                    dataset_sha256=hashlib.sha256((baseline_directory / "dataset.json").read_bytes()).hexdigest()),
               directory / "best.pt")
    monkeypatch.setattr(evaluation, "ExperimentData", lambda path: baseline)
    result = evaluation.evaluate_flow(synthetic_data, baseline_directory, seeds=(7,), noise_seeds=(11,),
                                      device="cpu", geometry=False)
    assert result["selection_split"] == "validation"
    assert result["selected_method"] in ("diffusion_vq", "diffusion_vae")
    assert {r["method"] for r in result["summaries"]} == {
        "pca", "interpolation", "legacy_diffusion", "diffusion_vq", "diffusion_vae"}
    assert len(result["summaries"]) == 10
    assert len(result["latent_metrics"]) == 16
    assert (synthetic_data.directory / "comparison.md").exists()
    baseline.split[0] = 2
    with pytest.raises(ValueError, match="split"):
        evaluation.validate_baseline(synthetic_data, baseline)


def test_training_cli_both_backends_smoke(synthetic_data, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/grasp_manifold.py"
    spec = importlib.util.spec_from_file_location("grasp_flow_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "ExperimentData", lambda path: synthetic_data)
    monkeypatch.setattr(sys, "argv", [str(script), "train", "--model", "diffusion", "--latent-kind", "both",
                                    "--steps", "2", "--min-steps", "0", "--batch-size", "8",
                                    "--validation-every", "1", "--hidden-dim", "32", "--latent-dim", "4",
                                    "--codebook-size", "8", "--flow-steps", "4", "--device", "cpu"])
    cli.main()
    registry = json.loads((synthetic_data.directory / "models.json").read_text())
    assert registry["default_diffusion"] == "diffusion_vq"
    runs = json.loads((synthetic_data.directory / "training_runs.json").read_text())
    assert {r["method"] for r in runs} == {"diffusion_vq", "diffusion_vae"}
    assert all(r["stages"]["flow"]["completed_steps"] == 2 for r in runs)


@pytest.mark.parametrize("kwargs", [dict(batch_size=3), dict(batch_size=7), dict(lr=float("nan")),
                                  dict(min_steps=-1), dict(sigma_min=.4), dict(codebook_weight=-1)])
def test_flow_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs).validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_loss_and_integration_smoke(synthetic_data):
    for kind in ("vq", "vae"):
        model = PoseFlowModel(kind, synthetic_data, small_config()).cuda()
        x = torch.tensor(synthetic_data.x[:4], dtype=torch.float32, device="cuda")
        groups = torch.tensor([0, 0, 1, 1], device="cuda")
        sizes = torch.ones(4, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(6)
        model.autoencoder_objective(x, groups, sizes, torch.tensor([1, 0, 3, 2], device="cuda"), gen)[0].backward()
        model.freeze_autoencoder()
        model.flow_objective(x, groups, sizes, gen).backward()
        assert torch.isfinite(model.denoise(x, groups, sizes)).all()
