#!/usr/bin/env python3
"""Audit the actual pose-flow input samplers without training any model.

Exports reproducible samples, quantitative diagnostics, PDF/PNG figures and
an offline interactive HTML. The uniform joint box is a mathematical control,
not a distribution of physically valid grasps or a training proposal.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from plotly.offline import get_plotlyjs
from scipy.spatial import cKDTree

from retarget.env.grasp.manifold import ExperimentData, tensors, write_json
from retarget.env.grasp.manifold_flow import (
    FlowConfig,
    finger_corruption,
    pair_table,
    sample_autoencoder_pairs,
)

COLORS = ("#0072B2", "#D55E00", "#7B8790", "#009E73")
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


def probability(values, bins):
    counts = np.bincount(np.asarray(values, dtype=int), minlength=bins)
    return counts / counts.sum()


def sampling_audit(data, n, seed, config):
    """Call the same pair sampler and corruption function as training."""
    if n < config.batch_size or n % config.batch_size:
        raise ValueError("samples must be a positive multiple of batch_size")
    config.validate()
    torch.set_num_threads(2)
    train_ids = np.flatnonzero(data.mask("train"))
    x, groups, sizes = tensors(data, train_ids, "cpu")
    table, counts, unique = pair_table(groups, sizes)
    labels = [r["template"] for r in data.meta["groups"]]
    type_names = sorted(set(labels))
    group_types = torch.tensor([type_names.index(label) for label in labels])
    if len(group_types[unique].unique()) < 2:
        raise ValueError("Need at least two grasp types for the stage-one sampler")
    stages = {}
    for offset, stage in enumerate(("autoencoder", "flow")):
        generator = torch.Generator().manual_seed(seed + offset * 100000)
        values = {key: [] for key in ("index", "observed", "noise", "rounds", "sigma_step", "mask", "time")}
        for _ in range(n // config.batch_size):
            if stage == "autoencoder":
                ids, partners = sample_autoencoder_pairs(table, counts, unique, group_types,
                                                        config.batch_size, generator)
                assert torch.equal(groups[ids], groups[ids][partners])
                assert (sizes[ids] != sizes[ids][partners]).all()
            else:
                ids = torch.randint(len(x), (config.batch_size,), generator=generator)
            observed, info = finger_corruption(x[ids], generator, max_rounds=config.max_rounds,
                                               sigma_min=config.sigma_min, sigma_max=config.sigma_max)
            times = torch.rand(len(ids), generator=generator)
            values["index"].append(train_ids[ids.numpy()])
            values["observed"].append(observed.numpy())
            for key in ("noise", "rounds", "sigma_step", "mask"):
                values[key].append(info[key].numpy())
            values["time"].append(times.numpy())
        stages[stage] = {key: np.concatenate(value) for key, value in values.items()}
    return stages


def measure(data, stages, uniform, config):
    train_ids = np.flatnonzero(data.mask("train"))
    clean = data.x[train_ids]
    tree = cKDTree(clean)
    uniform_dist = tree.query(uniform, workers=2)[0] / np.sqrt(20)
    unique_sizes = sorted(np.unique(data.size[train_ids]).tolist())
    expected_mask = np.array([1 / (6 * math.comb(5, int(mask).bit_count())) for mask in range(32)])
    all_results = {}
    for name, sample in stages.items():
        observed, noise, index = sample["observed"], sample["noise"], sample["index"]
        rounds, mask = sample["rounds"], sample["mask"]
        masks = (mask * (1 << np.arange(5))).sum(-1)
        counts = mask.sum(-1)
        gp = probability(data.group[index], len(data.subspaces))
        sigma_counts = np.histogram(sample["sigma_step"], np.linspace(config.sigma_min, config.sigma_max, 11))[0]
        distance = tree.query(observed, workers=2)[0] / np.sqrt(20)
        sample["nearest_distance"] = distance
        sample["masks"] = masks
        eig = np.linalg.eigvalsh(np.cov(noise, rowvar=False))
        spectrum = eig / eig.sum()
        variance = .5 * ((config.max_rounds + 1) / 2) * (
            config.sigma_min ** 2 + config.sigma_min * config.sigma_max + config.sigma_max ** 2) / 3
        nz = np.repeat(mask, 4, axis=1)
        sigma_coord = np.repeat(sample["sigma_step"][:, None] * np.sqrt(rounds), 4, axis=1)
        z = noise[nz] / sigma_coord[nz]
        per_joint = []
        hist = []
        for joint in range(20):
            values = observed[:, joint]
            h = np.histogram(values, np.linspace(-1, 1, 21))[0]
            hist.append(h / len(values))
            p = h / h.sum()
            per_joint.append(float(.5 * np.abs(p - 1 / 20).sum()))
        sample["joint_hist"] = np.array(hist)
        group_records = []
        for group, record in enumerate(data.meta["groups"]):
            available = train_ids[data.group[train_ids] == group]
            drawn = index[data.group[index] == group]
            group_records.append(dict(group=group, label=record["template"], primitive=record["primitive"],
                                      probability=float(gp[group]), train_nodes=len(available),
                                      nodes_seen=len(np.unique(drawn)),
                                      node_frequency_min=int(np.min(np.unique(drawn, return_counts=True)[1]))))
        expected_groups = (np.ones(len(gp)) / len(gp) if name == "autoencoder"
                           else probability(data.group[train_ids], len(gp)))
        all_results[name] = dict(
            samples=len(index), clean_nodes_seen=len(np.unique(index)), train_nodes=len(train_ids),
            grasp_groups_seen=int((gp > 0).sum()), group_records=group_records,
            group_probability_min=float(gp.min()), group_probability_max=float(gp.max()),
            group_probability_max_min_ratio=float(gp.max() / gp.min()),
            group_total_variation_from_sampler_target=float(.5 * np.abs(gp - expected_groups).sum()),
            noisy_finger_count_probabilities=probability(counts, 6).tolist(),
            active_finger_probabilities=mask.mean(0).tolist(),
            active_round_probabilities=probability(rounds[mask] - 1, config.max_rounds).tolist(),
            sigma_bin_probabilities=(sigma_counts / sigma_counts.sum()).tolist(),
            flow_time_bin_probabilities=(np.histogram(sample["time"], np.linspace(0, 1, 11))[0] / len(index)).tolist()
                if name == "flow" else None,
            mask_probabilities=probability(masks, 32).tolist(), mask_target_probabilities=expected_mask.tolist(),
            mask_total_variation_from_target=float(.5 * np.abs(probability(masks, 32) - expected_mask).sum()),
            masks_seen=len(np.unique(masks)),
            clean_input_fraction=float(np.mean(counts == 0)),
            exactly_zero_noise_coordinate_fraction=float(np.mean(noise == 0)),
            noise_covariance_rank=int(np.linalg.matrix_rank(np.cov(noise, rowvar=False))),
            noise_covariance_effective_rank=float(np.exp(-(spectrum * np.log(spectrum)).sum())),
            noise_variance_empirical=float(noise.var()), noise_variance_theoretical=float(variance),
            standardized_active_noise_mean=float(z.mean()), standardized_active_noise_std=float(z.std()),
            noise_rms_quantiles=dict(zip(("p50", "p90", "p99"), np.quantile(np.sqrt((noise ** 2).mean(-1)), [.5, .9, .99]).tolist())),
            nearest_clean_rmse_quantiles=dict(zip(("p50", "p90", "p99"), np.quantile(distance, [.5, .9, .99]).tolist())),
            fraction_within_clean_rmse={str(r): float(np.mean(distance <= r)) for r in (.05, .1, .2)},
            any_joint_limit_violation_fraction=float(np.mean((np.abs(observed) > 1).any(-1))),
            per_joint_mean_binned_tvd_from_uniform_box=float(np.mean(per_joint)),
            primitive_probabilities={primitive: float(np.mean([data.meta["groups"][g]["primitive"] == primitive
                                       for g in data.group[index]]))
                                     for primitive in sorted({r["primitive"] for r in data.meta["groups"]})},
            size_node_probabilities={str(s): float(np.mean(data.size[index] == s)) for s in unique_sizes})
        assert np.isfinite(observed).all() and np.all(sample["noise"][~nz] == 0)
        assert np.all(data.split[index] == 0)
    return dict(samples_per_stage=len(next(iter(stages.values()))["index"]),
                configuration=dict(sigma_min=config.sigma_min, sigma_max=config.sigma_max,
                                   max_rounds=config.max_rounds, batch_size=config.batch_size),
                stages=all_results, uniform_joint_box_control=dict(
                    samples=len(uniform), nearest_clean_rmse_quantiles=dict(zip(("p50", "p90", "p99"),
                    np.quantile(uniform_dist, [.5, .9, .99]).tolist())))), uniform_dist


def make_figures(data, stages, report, uniform, uniform_dist, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .15, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    a, b = stages["autoencoder"], stages["flow"]
    cfg = report["configuration"]
    summaries = report["stages"]
    for i, name in enumerate(stages):
        label, color, sample = ("Stage 1: encoder", COLORS[0], a) if i == 0 else ("Stage 2: flow", COLORS[1], b)
        s = summaries[name]
        axes[0, 0].bar(np.arange(6) + (i - .5) * .35, s["noisy_finger_count_probabilities"], width=.35, color=color, label=label)
        axes[0, 1].bar(np.arange(5) + (i - .5) * .35, s["active_finger_probabilities"], width=.35, color=color)
        axes[0, 2].plot(np.arange(1, cfg["max_rounds"] + 1), s["active_round_probabilities"], "o-", color=color)
        axes[0, 3].hist(sample["sigma_step"], bins=np.linspace(cfg["sigma_min"], cfg["sigma_max"], 21),
                        density=True, histtype="step", lw=1.5, color=color)
        axes[1, 0].plot(np.arange(len(data.subspaces)), [r["probability"] for r in s["group_records"]], "o-", color=color, ms=3)
        axes[1, 1].plot(np.arange(32), s["mask_probabilities"], ".-", color=color)
        noise_rms = np.sqrt((sample["noise"] ** 2).mean(-1))
        axes[1, 2].hist(noise_rms, bins=50, density=True, histtype="step", lw=1.5, color=color)
        values = np.sort(sample["nearest_distance"])
        axes[2, 2].plot(values, np.linspace(0, 1, len(values)), color=color, label=label)
    axes[0, 0].axhline(1/6, color=COLORS[2], ls="--"); axes[0, 0].legend()
    axes[0, 0].set(title="A  Number of noisy fingers", xlabel="Selected fingers (0-5)", ylabel="Probability", ylim=(0, .22))
    axes[0, 1].axhline(.5, color=COLORS[2], ls="--")
    axes[0, 1].set(xticks=np.arange(5), xticklabels=FINGERS, ylim=(0, .6), title="B  Finger selection symmetry", ylabel="Selection probability")
    axes[0, 2].axhline(1/cfg["max_rounds"], color=COLORS[2], ls="--")
    axes[0, 2].set(title="C  Rounds on selected fingers", xlabel="Rounds", ylabel="Conditional probability", ylim=(0, .27))
    axes[0, 3].axhline(1/(cfg["sigma_max"]-cfg["sigma_min"]), color=COLORS[2], ls="--")
    axes[0, 3].set(title="D  Per-round Gaussian scale", xlabel="Sigma in normalized joint units", ylabel="Density")
    axes[1, 0].axhline(1/len(data.subspaces), color=COLORS[2], ls="--")
    axes[1, 0].set(title="E  Grasp-type exposure differs by stage", xlabel="Grasp group (dataset order)", ylabel="Sample probability")
    axes[1, 1].plot(np.arange(32), summaries["flow"]["mask_target_probabilities"], "--", color=COLORS[2], label="Specified mixture")
    axes[1, 1].set(title="F  Finger subsets are NOT equiprobable", xlabel="5-bit finger mask (0-31)", ylabel="Probability")
    axes[1, 2].set(title="G  Local noise magnitude + clean mass", xlabel="Noise RMSE across 20 joints", ylabel="Density")
    primitives = list(summaries["flow"]["primitive_probabilities"])
    for i, name in enumerate(stages):
        axes[1, 3].bar(np.arange(len(primitives)) + (i - .5) * .35,
                        [summaries[name]["primitive_probabilities"][p] for p in primitives], width=.35, color=COLORS[i])
    axes[1, 3].set(title="H  Primitive types are not balanced", xticks=np.arange(len(primitives)), xticklabels=primitives, ylabel="Probability")
    clean = data.x[data.mask("train")]
    center = clean.mean(0)
    _, singular, vt = np.linalg.svd(clean-center, full_matrices=False)
    explained = singular ** 2 / (singular ** 2).sum()
    rng = np.random.default_rng(11)
    choose = rng.choice(len(b["observed"]), min(8000, len(b["observed"])), replace=False)
    pc = (b["observed"][choose] - center) @ vt[:3].T
    cp = (clean - center) @ vt[:3].T
    up = (uniform[:8000] - center) @ vt[:3].T
    axes[2, 0].scatter(up[:, 0], up[:, 1], s=2, c=COLORS[2], alpha=.12, rasterized=True, label="Uniform box control")
    axes[2, 0].scatter(pc[:, 0], pc[:, 1], s=2, c=COLORS[1], alpha=.15, rasterized=True, label="Flow input")
    axes[2, 0].scatter(cp[:, 0], cp[:, 1], s=6, c=COLORS[0], label="Clean training poses")
    axes[2, 0].set(title=f"I  Fixed clean PCA: first 2 = {explained[:2].sum():.1%}", xlabel="PC1", ylabel="PC2")
    axes[2, 0].legend(markerscale=2, loc="upper right")
    heatmap = axes[2, 1].imshow(b["joint_hist"], aspect="auto", origin="lower", extent=(-1, 1, -.5, 19.5), cmap="Blues")
    axes[2, 1].set(title="J  Joint marginals: concentrated, not flat", xlabel="Normalized joint angle", ylabel="Joint index (0-19)")
    axes[2, 1].grid(False); fig.colorbar(heatmap, ax=axes[2, 1], label="Probability per bin", shrink=.8)
    values = np.sort(uniform_dist)
    axes[2, 2].plot(values, np.linspace(0, 1, len(values)), color=COLORS[2], ls="--", label="Uniform box control")
    axes[2, 2].set(title="K  Distance to nearest clean pose", xlabel="Nearest-reference RMSE (20D)", ylabel="Cumulative probability")
    axes[2, 2].legend(loc="lower right")
    inside = summaries["flow"]["fraction_within_clean_rmse"]["0.1"]
    axes[2, 3].axis("off")
    text = (f"{len(b['index']):,} samples / stage\n"
            f"{summaries['flow']['clean_nodes_seen']} / {len(clean)} training nodes sampled\n"
            f"{summaries['flow']['masks_seen']} / 32 finger masks observed\n"
            f"{summaries['flow']['clean_input_fraction']:.1%} entirely clean observations\n"
            f"{inside:.1%} within RMSE 0.1 of a clean pose\n"
            f"Noise covariance rank: {summaries['flow']['noise_covariance_rank']} / 20\n\n"
            "Balanced corruption choices;\ncontinuous LOCAL pose diversity.\n\n"
            "NOT uniform in the 20D joint box.\nFull rank does not imply uniformity.\nNo learned latent/generation claim.")
    axes[2, 3].text(.02, .98, text, va="top", linespacing=1.5, fontsize=11)
    fig.suptitle("Pose-flow training-input sampling audit | no model training", fontsize=19, weight="bold")
    fig.savefig(output / "sampling_distribution.png", dpi=300)
    fig.savefig(output / "sampling_distribution.pdf")
    plt.close(fig)
    report["clean_pca_explained_variance"] = explained.tolist()
    return center, vt[:3]


def interactive(data, stages, uniform, center, basis, report, output, seed):
    rng = np.random.default_rng(seed)
    payload = dict(stages={}, groups=data.meta["groups"], explained=report["clean_pca_explained_variance"][:3])
    for name, sample in stages.items():
        indices = rng.choice(len(sample["index"]), min(10000, len(sample["index"])), replace=False)
        q = sample["observed"][indices]
        payload["stages"][name] = dict(q=q.round(5).tolist(), pc=((q-center) @ basis.T).round(5).tolist(),
                                      group=data.group[sample["index"][indices]].tolist(),
                                      k=sample["mask"][indices].sum(-1).tolist(),
                                      sigma=sample["sigma_step"][indices].round(5).tolist(),
                                      size=data.size[sample["index"][indices]].tolist())
    clean = data.x[data.mask("train")]
    payload["clean"] = dict(q=clean.round(5).tolist(), pc=((clean-center) @ basis.T).round(5).tolist(),
                            group=data.group[data.mask("train")].tolist())
    u = uniform[:5000]
    payload["uniform"] = dict(q=u.round(5).tolist(), pc=((u-center) @ basis.T).round(5).tolist())
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Pose Flow · 采样分布检查</title>
<style>body{margin:24px;background:#f4f7f8;color:#243b47;font:15px/1.6 system-ui}h1{margin-bottom:4px}p{max-width:1100px}.controls{display:flex;gap:20px;flex-wrap:wrap;background:white;padding:16px;border-radius:8px}label{display:flex;flex-direction:column}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.plot{height:560px;background:white;margin-top:16px}img{max-width:100%;background:white}a{color:#0072b2}select{padding:6px}#status{font-family:monospace}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>采样足够多样吗？局部覆盖与全空间均匀性</h1>
<p>直接调用训练采样器，没有训练网络。各阶段统计使用全部样本；交互散点每阶段随机显示最多 10,000 个。PCA 仅从干净训练姿态拟合，两阶段共用坐标轴。二维／三维投影不能证明 20 维全空间覆盖。</p>
<div class="controls"><label>训练阶段<select id="phase"><option value="flow">第二阶段 · Flow</option><option value="autoencoder">第一阶段 · Encoder</option></select></label>
<label>抓型<select id="group"><option value="all">全部抓型</option></select></label>
<label>颜色<select id="color"><option value="k">加噪手指数</option><option value="group">抓型</option><option value="sigma">单轮 sigma</option><option value="size">尺寸倍率</option></select></label>
<label>关节横轴<select id="x"></select></label><label>关节纵轴<select id="y"></select></label>
<label>数学对照<input id="uniform" type="checkbox">显示均匀关节盒（不代表有效抓姿）</label></div>
<p id="status"></p><div class="grid"><div class="plot" id="joints"></div><div class="plot" id="pca"></div></div>
<h2>全部样本的统计</h2><p>0–5 根手指的数量均匀采样，不等于 32 种手指组合均匀采样。高斯扰动保留局部密度，且约 1/6 输入完全干净，因此不会均匀覆盖关节空间。随机均匀关节盒仅作几何对照，未做碰撞或抓握有效性筛选。</p>
<img src="sampling_distribution.png" alt="Sampling distribution: 12 diagnostic panels"><p><a href="sampling_distribution.pdf">PDF</a> · <a href="sampling_metrics.json">完整统计 JSON</a> · <a href="sampling_samples.npz">采样 NPZ</a> · <a href="report.md">结论与复现说明</a></p>
<script>__PLOTLY__</script><script>const DATA=__DATA__;
const $=id=>document.getElementById(id);
DATA.groups.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=g.template+' / '+g.primitive;$('group').append(o)});
for(const axis of ['x','y'])for(let i=0;i<20;i++){const o=document.createElement('option');o.value=i;o.textContent=['拇指','食指','中指','无名指','小指'][Math.floor(i/4)]+' · 关节 '+(i%4+1);$(axis).append(o)}$('y').value=1;
function draw(){const s=DATA.stages[$('phase').value],g=$('group').value,idx=s.group.map((_,i)=>i).filter(i=>g==='all'||s.group[i]===+g),ci=DATA.clean.group.map((_,i)=>i).filter(i=>g==='all'||DATA.clean.group[i]===+g),x=+$('x').value,y=+$('y').value,c=$('color').value,marker={size:3,color:idx.map(i=>s[c][i]),colorscale:'Viridis',opacity:.35,colorbar:{title:c}};
const tr=[{type:'scattergl',mode:'markers',name:'带噪输入',x:idx.map(i=>s.q[i][x]),y:idx.map(i=>s.q[i][y]),marker,customdata:idx.map(i=>[s.group[i],s.k[i],s.sigma[i],s.size[i]]),hovertemplate:'x=%{x}<br>y=%{y}<br>group=%{customdata[0]}<br>fingers=%{customdata[1]}<br>sigma=%{customdata[2]}<br>size=%{customdata[3]}<extra></extra>'},{type:'scattergl',mode:'markers',name:'干净训练节点',x:ci.map(i=>DATA.clean.q[i][x]),y:ci.map(i=>DATA.clean.q[i][y]),marker:{color:'#0072B2',size:7,symbol:'cross'}}];
const pc=[{type:'scatter3d',mode:'markers',name:'带噪输入',x:idx.map(i=>s.pc[i][0]),y:idx.map(i=>s.pc[i][1]),z:idx.map(i=>s.pc[i][2]),marker:{...marker,size:2}},{type:'scatter3d',mode:'markers',name:'干净训练节点',x:ci.map(i=>DATA.clean.pc[i][0]),y:ci.map(i=>DATA.clean.pc[i][1]),z:ci.map(i=>DATA.clean.pc[i][2]),marker:{color:'#0072B2',size:4,symbol:'cross'}}];
if($('uniform').checked){const u=DATA.uniform;tr.unshift({type:'scattergl',mode:'markers',name:'均匀关节盒',x:u.q.map(q=>q[x]),y:u.q.map(q=>q[y]),marker:{color:'#8999a1',size:2,opacity:.12}});pc.unshift({type:'scatter3d',mode:'markers',name:'均匀关节盒',x:u.pc.map(q=>q[0]),y:u.pc.map(q=>q[1]),z:u.pc.map(q=>q[2]),marker:{color:'#8999a1',size:1,opacity:.08}})}
Plotly.react('joints',tr,{title:'原始归一化关节空间 · 可更换维度',xaxis:{title:$('x').selectedOptions[0].text},yaxis:{title:$('y').selectedOptions[0].text},margin:{l:65,r:40,t:55,b:60},legend:{orientation:'h'},uirevision:'joints'},{responsive:true,displaylogo:false});
Plotly.react('pca',pc,{title:'固定 PCA 投影 · 前三维方差 '+(100*DATA.explained.reduce((a,b)=>a+b,0)).toFixed(1)+'%',scene:{xaxis:{title:'PC1'},yaxis:{title:'PC2'},zaxis:{title:'PC3'}},margin:{l:0,r:10,t:55,b:0},legend:{orientation:'h'},uirevision:'pca'},{responsive:true,displaylogo:false});
$('status').textContent='当前显示 '+idx.length+' 个带噪样本 / '+ci.length+' 个干净训练节点；散点子样本不用于估计总体概率。'}
for(const id of ['phase','group','color','x','y','uniform'])$(id).onchange=draw;draw();</script></html>'''
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (output / "index.html").write_text(html.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", encoded))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "results/grasp_manifold_20261103")
    parser.add_argument("--output", type=Path, default=ROOT / "output/grasp_pose_flow_sampling_broad_20261103")
    parser.add_argument("--samples", type=int, default=102400, help="per stage; multiple of batch-size")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--sigma-min", type=float, default=.01)
    parser.add_argument("--sigma-max", type=float, default=FlowConfig().sigma_max)
    args = parser.parse_args()
    config = FlowConfig(batch_size=args.batch_size, max_rounds=args.max_rounds,
                        sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    data = ExperimentData(args.experiment)
    print(f"Sampling {args.samples:,} observations per stage; no model training", flush=True)
    stages = sampling_audit(data, args.samples, args.seed, config)
    uniform = np.random.default_rng(args.seed).uniform(-1, 1, (args.samples, 20))
    report, uniform_dist = measure(data, stages, uniform, config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    center, basis = make_figures(data, stages, report, uniform, uniform_dist, output)
    report.update(seed=args.seed, experiment=str(data.directory),
                  no_model_training=True, source_sha256=data.meta["source_sha256"],
                  interpretation="Diverse local corruption with balanced noise choices; not uniform 20D pose coverage",
                  limitations=["Training-input distribution only, not learned latent or generated pose distribution",
                               "No collision/contact validity test on noisy samples",
                               "PCA and marginal plots cannot establish full-dimensional uniformity"])
    write_json(output / "sampling_metrics.json", report)
    arrays = {f"{stage}_{key}": value for stage, values in stages.items() for key, value in values.items()}
    np.savez_compressed(output / "sampling_samples.npz", **arrays,
                        uniform_control=uniform.astype(np.float32), pca_center=center, pca_basis=basis)
    interactive(data, stages, uniform, center, basis, report, output, args.seed)
    flow = report["stages"]["flow"]
    lines = ["# Pose flow 采样分布检查", "", "只调用真实训练输入采样器与加噪函数，没有训练模型。", "",
             f"每阶段 {args.samples:,} 个样本；共 {2*args.samples:,} 个带噪训练输入。",
             f"第二阶段覆盖 {flow['clean_nodes_seen']}/{flow['train_nodes']} 个训练尺寸节点、{flow['grasp_groups_seen']} 个抓型、{flow['masks_seen']}/32 种手指组合。",
             "", "| 检查项 | 第一阶段 | 第二阶段 |", "|---|---:|---:|"]
    for label, key, fmt in (("抓型概率最大/最小", "group_probability_max_min_ratio", ".3f"),
                            ("完全干净输入", "clean_input_fraction", ".2%"),
                            ("噪声坐标恰为零", "exactly_zero_noise_coordinate_fraction", ".2%"),
                            ("噪声协方差有效维数 / 20", "noise_covariance_effective_rank", ".3f"),
                            ("存在关节越限的输入", "any_joint_limit_violation_fraction", ".2%")):
        lines.append(f"| {label} | {format(report['stages']['autoencoder'][key], fmt)} | {format(flow[key], fmt)} |")
    lines.extend(("", f"第二阶段最近干净训练姿态距离（20 维 RMSE）中位数 {flow['nearest_clean_rmse_quantiles']['p50']:.4f}，"
                  f"99% 分位数 {flow['nearest_clean_rmse_quantiles']['p99']:.4f}；"
                  f"均匀关节盒对照中位数 {report['uniform_joint_box_control']['nearest_clean_rmse_quantiles']['p50']:.4f}。",
                  "", "结论：噪声选择接近设计分布，方向覆盖 20 维，但样本围绕有限标准抓姿聚集，属于局部多样性；不能视为均匀覆盖整个关节空间，也不能据此证明模型已学得多样 latent。",
                  "", "0–5 根手指数量等概率，32 种具体组合不等概率：全不选和全选各 1/6，任一指定两指或三指组合各 1/60。第一阶段按抓型平衡，第二阶段按训练节点均匀，因此节点较多的抓型更常出现。primitive 和尺寸倍率也不强制等概率。",
                  "", "图中均匀关节盒只作为数学对照，并不代表物理有效抓姿。加噪没有裁剪，越限比例如上；没有评估碰撞或接触有效性。",
                  "", "复现：", "```bash", f".venv-contact/bin/python scripts/inspect_pose_flow_sampling.py --samples {args.samples} --seed {args.seed}", "```"))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(output=str(output), flow=flow, uniform=report["uniform_joint_box_control"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
