"""Mathematical contracts and leakage checks for the standalone experiment."""
from types import SimpleNamespace
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from retarget.env.grasp.manifold import (DenoisingModel, SizeCurve, Subspace, TrainConfig,
                                         ExperimentData, Predictor, alpha_schedule, sigma_timestep, split_size_nodes)


def synthetic_space():
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    x = .2 + rng.normal(size=(80, 3)) @ basis.T
    return Subspace.fit(x)


def test_decomposition_is_exact_orthogonal_and_sign_invariant():
    space = synthetic_space()
    x = np.random.default_rng(8).normal(size=(10, 20))
    principal, residual = space.decompose(x)
    np.testing.assert_allclose(principal + residual, x, atol=1e-14)
    np.testing.assert_allclose(residual @ space.basis, 0, atol=1e-14)
    original = principal.copy()
    space.basis *= -1
    np.testing.assert_allclose(space.project(x), original, atol=1e-14)
    assert space.numerical_rank == 3
    assert space.retained_fraction >= .95


def test_single_template_noise_has_no_privileged_grasp_axis():
    rng = np.random.default_rng(13)
    template = rng.normal(size=20)
    noisy = template + .1 * rng.normal(size=(20000, 20))
    space = Subspace.fit(noisy)
    assert space.numerical_rank == 20
    assert space.basis.shape == (20, 19)  # explicit non-full subspace cap
    np.testing.assert_allclose(space.mean, template, atol=.003)
    assert space.eigenvalues.max() / space.eigenvalues.min() < 1.2


def test_constant_class_has_zero_dimensional_subspace_and_serializes():
    x = np.ones((10, 20))
    space = Subspace.from_record(Subspace.fit(x).record())
    assert space.basis.shape == (20, 0)
    np.testing.assert_allclose(space.project(x + 2), x)


def test_size_node_split_precedes_augmentation_and_keeps_endpoints():
    groups = np.repeat(np.arange(3), 17 * 4)
    sizes = np.tile(np.repeat(np.linspace(.5, 1.5, 17), 4), 3)
    split = split_size_nodes(groups, sizes, seed=7)
    np.testing.assert_array_equal(split, split_size_nodes(groups, sizes, seed=7))
    for group in range(3):
        for size in np.unique(sizes):
            assert len(np.unique(split[(groups == group) & (sizes == size)])) == 1
        assert set(split[groups == group]) == {0, 1, 2}
    assert np.all(split[np.isin(sizes, [.5, 1.5])] == 0)
    with pytest.raises(ValueError, match="at least five"):
        split_size_nodes([0, 0, 0], [1., 2., 3.])


def test_size_tightening_preserves_residual_and_zero_is_identity():
    space = synthetic_space()
    sizes = np.linspace(.5, 1.5, 8)
    poses = space.mean + sizes[:, None] * space.basis[:, 0]
    curve = SizeCurve(sizes, poses)
    observed = np.random.default_rng(8).normal(size=20)
    principal, residual = space.decompose(observed)
    result, target, delta = curve.tighten(principal, residual, 1., 0.)
    np.testing.assert_allclose(result, observed, atol=1e-14)
    np.testing.assert_array_equal(delta, np.zeros(20))
    result, target, delta = curve.tighten(principal, residual, 1., .1)
    np.testing.assert_allclose(space.decompose(result)[1], residual, atol=1e-14)
    assert target == pytest.approx(.9)
    assert curve.tighten(principal, residual, .5, .2)[1] == .5
    np.testing.assert_allclose(curve(10), curve(1.5))
    with pytest.raises(ValueError):
        curve.tighten(principal, residual, 1., -.1)


def test_multidimensional_size_curve_interpolates_coefficients_inside_subspace():
    space = synthetic_space()
    sizes = np.linspace(.5, 1.5, 8)
    coefficients = np.stack((sizes, np.sin(4*sizes), sizes**3), -1)
    poses = space.mean + coefficients @ space.basis.T
    curve = SizeCurve(sizes, poses, space)
    interpolated = curve(np.linspace(.5, 1.5, 137))
    np.testing.assert_allclose(space.project(interpolated), interpolated, atol=1e-14)
    observed = np.random.default_rng(8).normal(size=20)
    principal, residual = space.decompose(observed)
    result, _, _ = curve.tighten(principal, residual, 1., .13)
    np.testing.assert_allclose(space.decompose(result)[1], residual, atol=1e-14)


def test_residual_retains_only_perpendicular_part_of_known_bias_and_noise():
    space = synthetic_space()
    rng = np.random.default_rng(22)
    clean = space.project(rng.normal(size=20))
    bias, noise = rng.normal(size=(2, 20))
    _, residual = space.decompose(clean + bias + noise)
    expected = bias + noise - (bias + noise) @ space.basis @ space.basis.T
    np.testing.assert_allclose(residual, expected, atol=1e-14)
    assert np.linalg.norm(residual-bias) > .1


def model_data():
    return SimpleNamespace(subspaces=[synthetic_space()], log_center=0., log_scale=1.)


def test_dictionary_has_128_atoms_and_output_stays_in_class_space():
    torch.manual_seed(7)
    data = model_data()
    model = DenoisingModel("dictionary", data, TrainConfig())
    assert model.dictionary.shape == (20, 128)
    assert torch.linalg.matrix_rank(model.dictionary) <= 20
    x, g, s = torch.randn(6, 20), torch.zeros(6, dtype=torch.long), torch.ones(6)
    out = model.denoise(x, g, s, 20)
    np.testing.assert_allclose(out.numpy(), data.subspaces[0].project(out.numpy()), atol=2e-7)
    loss = model.objective(x, g, s, torch.Generator().manual_seed(2))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dictionary.grad is not None


class OracleEpsilon(nn.Module):
    def __init__(self, clean, alphas):
        super().__init__()
        self.clean, self.alphas = clean, alphas

    def forward(self, x, groups, sizes, times):
        index = torch.round(times * (len(self.alphas) - 1)).long()
        a = self.alphas[index, None]
        return (x-a.sqrt()*self.clean)/(1-a).sqrt()


@pytest.mark.parametrize("timestep", [1, 20, 500, 1000])
def test_ddim_starts_from_observation_and_matches_oracle(timestep):
    model = DenoisingModel("diffusion", model_data(), TrainConfig())
    groups, sizes = torch.zeros(4, dtype=torch.long), torch.ones(4)
    clean = model.project(torch.randn(4, 20), groups)
    model.encoder = OracleEpsilon(clean, model.alphas)
    sigma = torch.sqrt((1-model.alphas[timestep])/model.alphas[timestep])
    observed = clean + sigma*torch.randn_like(clean)
    predicted = model.denoise(observed, groups, sizes, timestep)
    torch.testing.assert_close(predicted, clean, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(model.denoise(observed, groups, sizes, timestep), predicted, atol=0, rtol=0)


def test_noise_schedule_has_exact_clean_origin_and_monotone_noise():
    schedule = alpha_schedule()
    assert schedule[0] == 1.
    assert np.all(np.diff(schedule) < 0)
    assert sigma_timestep(0., schedule) == 0
    assert sigma_timestep(.03, schedule) < sigma_timestep(.3, schedule)
    for sigma in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            sigma_timestep(sigma, schedule)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required for offline JavaScript FK check")
def test_offline_viewer_fk_matches_native_model():
    from retarget.env.grasp.manifold_report import HTML, visual_model
    from retarget.viewer.snapshot import GraspSnapshot
    import mujoco

    source = Path(__file__).resolve().parents[1] / "results/grasp_current_20261103"
    if not (source / "manifest.json").exists():
        pytest.skip("Shared grasp snapshot not installed")
    snapshot = GraspSnapshot(source)
    model = snapshot.model(0)
    state = mujoco.MjData(model)
    q = snapshot.poses["qpos"][0].copy()
    q[6:] += .013
    state.qpos[:] = q
    mujoco.mj_kinematics(model, state)
    native = [state.xpos[model.body("right_palm_link").id].copy()]
    for finger, name in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
        native.extend(state.xpos[model.body(f"right_finger{finger}_link{k}").id].copy() for k in (2, 3, 4))
        native.append(state.site_xpos[model.site(f"right_{name}_tip").id].copy())
    exported = visual_model(model, {}, "test")
    code = "const add=" + HTML.split("const add=")[1].split("function traces(")[0]
    code += "\nconst m=" + json.dumps(exported) + ";"
    code += "console.log(JSON.stringify(jointPoints(m,frames(m," + json.dumps(q.tolist()) + "))));"
    result = subprocess.run(["node"], input=code, text=True, capture_output=True, check=True)
    np.testing.assert_allclose(json.loads(result.stdout), native, rtol=0, atol=1e-10)


def test_compact_bundle_loads_retained_models_and_preserves_pca_results():
    import hashlib

    root = Path(__file__).resolve().parents[1] / "results/grasp_manifold_20261103"
    if not root.exists():
        pytest.skip("Compact trained results not installed")
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    data = ExperimentData(root)
    assert not Path(data.meta["source"]).is_absolute()
    assert data.model_seed("dictionary") == 17
    assert data.model_seed("diffusion") == 27
    assert data.model_seed("dictionary", 23) == 23
    with np.load(root / "clean_poses.npz", allow_pickle=False) as poses:
        np.testing.assert_array_equal(poses["reference_qpos"], data.qpos)
        expected_pca = data.denormalize(data.project(data.x, data.group))
        np.testing.assert_allclose(poses["pca_qpos"][:, 6:], expected_pca, atol=1e-14)
        for method in ("dictionary", "diffusion"):
            predictor = Predictor(data, method, device="cpu")
            result = predictor.predict(data.x[:4], data.group[:4], data.size[:4], sigma=.1)
            assert result.shape == (4, 20) and np.isfinite(result).all()
            np.testing.assert_allclose(data.project(result, data.group[:4]), result, atol=2e-7)
    summary = json.loads((root / "evaluation.json").read_text())
    assert len(summary["completed_training_runs"]) == 6
    assert all(r["completed_steps"] == 3000 for r in summary["completed_training_runs"])


def test_compact_dataset_source_is_relative_to_bundle_not_working_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1] / "results/grasp_manifold_20261103"
    if not root.exists():
        pytest.skip("Compact trained results not installed")
    bundle = tmp_path / "results" / root.name
    source = bundle.parent / "grasp_current_20261103"
    bundle.mkdir(parents=True); source.mkdir()
    for name in ("dataset.json", "dataset.npz"):
        shutil.copy2(root / name, bundle / name)
    metadata = json.loads((root / "dataset.json").read_text())
    for name in metadata["source_sha256"]:
        shutil.copy2(root.parent / "grasp_current_20261103" / name, source / name)
    monkeypatch.chdir(tmp_path)
    copied = ExperimentData(bundle)
    assert copied.source == source
    assert copied.qpos.shape == (640, 26)
