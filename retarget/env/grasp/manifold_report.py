"""Self-contained offline 3D report, including source meshes and browser FK."""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np

from .manifold import METHODS, PoseChecker, Predictor, sigma_timestep, write_json

STAGES = ("truth", "observed", "projection", "clean", "tight_clean", "recombined", "clipped", "pca_recombined")


def visual_model(model, mesh_library, prefix):
    """Export native body/joint transforms; do not approximate hand FK in 3D plots."""
    bodies = []
    for i in range(model.nbody):
        joints = []
        start = int(model.body_jntadr[i])
        for j in range(start, start + int(model.body_jntnum[i])):
            if model.jnt_type[j] not in (2, 3):
                raise ValueError("Offline FK supports slide and hinge joints only")
            joints.append(dict(type=int(model.jnt_type[j]), axis=model.jnt_axis[j].tolist(),
                               pivot=model.jnt_pos[j].tolist(), adr=int(model.jnt_qposadr[j]),
                               ref=float(model.qpos0[model.jnt_qposadr[j]])))
        bodies.append(dict(parent=int(model.body_parentid[i]), p=model.body_pos[i].tolist(),
                           quat=model.body_quat[i].tolist(), joints=joints))
    geoms, seen = [], set()
    obj = model.body("grasp_object").id
    for i in range(model.ngeom):
        if int(model.geom_type[i]) != 7:  # mjGEOM_MESH
            continue
        mesh = int(model.geom_dataid[i])
        body = int(model.geom_bodyid[i])
        key = (body, mesh, *model.geom_pos[i], *model.geom_quat[i])
        if key in seen:
            continue
        seen.add(key)
        name = f"{prefix}_object" if body == obj else f"hand_{model.mesh(mesh).name}"
        if name not in mesh_library:
            v, n = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            f, k = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            mesh_library[name] = dict(v=model.mesh_vert[v:v+n].round(8).tolist(),
                                      f=model.mesh_face[f:f+k].tolist())
        geoms.append(dict(body=body, mesh=name, p=model.geom_pos[i].tolist(),
                          quat=model.geom_quat[i].tolist(), object=body == obj))
    tips = [dict(body=int(model.site(f"right_{name}_tip").bodyid[0]),
                 p=model.site(f"right_{name}_tip").pos.tolist())
            for name in ("thumb", "index", "middle", "ring", "pinky")]
    chain = [[model.body("right_palm_link").id,
              *[model.body(f"right_finger{i}_link{k}").id for k in (2, 3, 4)]] for i in range(1, 6)]
    return dict(bodies=bodies, body_names=[model.body(i).name for i in range(model.nbody)],
                geoms=geoms, tips=tips, chains=chain)


def demo(data, methods=METHODS, seed=None, noise_seed=11, sigmas=(0., .03, .1),
         fractions=(0., .1, .2), grasp_type=None, device="auto"):
    if not sigmas or not fractions or any(not np.isfinite(s) or s < 0 for s in sigmas):
        raise ValueError("Demo requires nonnegative finite noise and tightening lists")
    if any(not np.isfinite(f) or not 0 <= f < 1 for f in fractions):
        raise ValueError("Tightening fractions must be in [0,1)")
    report_dir = data.directory / "demo"
    report_dir.mkdir(parents=True, exist_ok=True)
    predictors = {method: Predictor(data, method, seed, device) for method in methods}
    rng = np.random.default_rng(7 if seed is None else seed)
    groups = [g for g, record in enumerate(data.meta["groups"])
              if grasp_type is None or int(record["template"].split("_")[0]) == grasp_type]
    if not groups:
        raise ValueError("No matching grasp type in the prepared dataset")
    meshes, models, examples, pose_arrays = {}, {}, [], []
    for group in groups:
        ids = np.flatnonzero((data.group == group) & data.mask("test"))
        index = int(rng.choice(ids))
        checker = PoseChecker(data, index)
        models[str(group)] = visual_model(checker.model, meshes, str(group))
        models[str(group)]["template"] = checker.template
        models[str(group)]["label"] = data.meta["groups"][group]["template"]
        models[str(group)]["primitive"] = data.meta["groups"][group]["primitive"]
        models[str(group)]["source_index"] = int(data.source_index[index])
        models[str(group)]["source_quality"] = data.meta["quality"][index]
        x, size, subspace = data.x[index], float(data.size[index]), data.subspaces[group]
        local_rng = np.random.default_rng(noise_seed + int(data.source_index[index]))
        direction = local_rng.standard_normal(20)
        perpendicular = direction - direction @ subspace.basis @ subspace.basis.T
        biases = dict(none=np.zeros(20),
                      orthogonal=.04 * perpendicular / max(np.sqrt(np.mean(perpendicular**2)), 1e-12),
                      general=.04 * direction / np.sqrt(np.mean(direction**2)))
        noise = local_rng.standard_normal(20)
        metric_cache = {}

        def inspect(normalized):
            q = data.qpos[index].copy()
            q[6:] = data.denormalize(normalized)
            key = q.tobytes()
            if key not in metric_cache:
                metric_cache[key] = checker.inspect(q)[0]
            return q, metric_cache[key]

        for bias_name, bias in biases.items():
            for sigma_requested in sigmas:
                schedule = next(iter(predictors.values())).schedule
                t = sigma_timestep(sigma_requested, schedule)
                sigma = math.sqrt((1 - schedule[t]) / schedule[t])
                observed = x + bias + sigma * noise
                principal, residual = subspace.decompose(observed)
                _, reference_residual = subspace.decompose(x)
                bias_perp = bias - bias @ subspace.basis @ subspace.basis.T
                noise_perp = sigma * (noise - noise @ subspace.basis @ subspace.basis.T)
                for method, predictor in predictors.items():
                    clean = predictor.predict(observed[None], [group], [size], sigma)[0]
                    for fraction in fractions:
                        recombined, target, delta = predictor.curves[group].tighten(clean, residual, size, fraction)
                        pca_recombined, _, _ = data.curve(group).tighten(principal, residual, size, fraction)
                        values = dict(truth=x, observed=observed, projection=principal, clean=clean,
                                      tight_clean=clean+delta, recombined=recombined,
                                      clipped=np.clip(recombined, -1., 1.), pca_recombined=pca_recombined)
                        poses, metrics = {}, {}
                        for stage in STAGES:
                            q, metrics[stage] = inspect(values[stage])
                            poses[stage] = q.round(10).tolist()
                        clipped_residual = subspace.decompose(values["clipped"])[1]
                        item = dict(group=group, index=index, method=method, bias=bias_name,
                            sigma_requested=float(sigma_requested), sigma_actual=sigma,
                            fraction=float(fraction), source_size=size, target_size=target,
                            size_axis=data.get_snapshot().manifest["cases"][int(data.source_index[index])]["primitive"]["size_axis"],
                            poses=poses, metrics=metrics,
                            known_bias=bias.tolist(), gaussian_noise=(sigma*noise).tolist(), residual=residual.tolist(),
                            reference_residual=reference_residual.tolist(), bias_perpendicular=bias_perp.tolist(),
                            noise_perpendicular=noise_perp.tolist(),
                            decomposition_max_error=float(np.max(np.abs(principal+residual-observed))),
                            residual_orthogonality_error=float(np.linalg.norm(residual @ subspace.basis)),
                            residual_vs_known_bias_rmse=float(np.sqrt(np.mean((residual-bias)**2))),
                            residual_increment_vs_bias_rmse=float(np.sqrt(np.mean((residual-reference_residual-bias)**2))),
                            retained_noise_rms=float(np.sqrt(np.mean(noise_perp**2))),
                            clipping_pose_change_rad=float(np.linalg.norm((values["clipped"]-recombined)*data.half)),
                            clipping_residual_change=float(np.linalg.norm(clipped_residual-residual)),
                            pure_pca_zero_tightening_error=float(np.max(np.abs(pca_recombined-observed))) if fraction == 0 else None)
                        examples.append(item)
                        pose_arrays.append(np.stack([data.qpos[index].copy() for _ in STAGES]))
                        for stage_index, stage in enumerate(STAGES):
                            pose_arrays[-1][stage_index, 6:] = data.denormalize(values[stage])
        print(f"demo: {data.meta['groups'][group]['template']} ({len(examples)} comparisons)", flush=True)
    np.savez_compressed(report_dir / "poses.npz", qpos=np.stack(pose_arrays), stages=np.array(STAGES),
                        groups=[e["group"] for e in examples], indices=[e["index"] for e in examples],
                        methods=[e["method"] for e in examples], biases=[e["bias"] for e in examples],
                        sigmas=[e["sigma_actual"] for e in examples], fractions=[e["fraction"] for e in examples])
    evaluation_file = data.directory / "evaluation.json"
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else None
    summaries = evaluation["summaries"] if evaluation else []
    payload = dict(groups=groups, models=models, meshes=meshes, examples=examples, methods=list(methods),
                   sigmas=list(sigmas), fractions=list(fractions), summaries=summaries,
                   selected_method=evaluation["selected_method"] if evaluation else methods[0],
                   model_seed=seed, model_seeds={m: data.model_seed(m, seed) for m in methods}, noise_seed=noise_seed)
    write_json(report_dir / "comparisons.json", {k: v for k, v in payload.items() if k not in ("models", "meshes")})
    from plotly.offline import get_plotlyjs
    html = HTML.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    (report_dir / "index.html").write_text(html)
    preview(data, payload, report_dir / "preview.png")
    return report_dir / "index.html"


def preview(data, payload, output):
    """Export a scientific figure using native FK, independent of browser drawing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    shown_method = payload["selected_method"] if payload["selected_method"] in payload["methods"] else payload["methods"][0]
    candidates = [e for e in payload["examples"] if e["method"] == shown_method
                  and e["sigma_requested"] > 0 and e["fraction"] > 0 and e["bias"] == "orthogonal"]
    if not candidates:
        candidates = payload["examples"]
    # Prefer the interpretable two-finger pinch when the full taxonomy is present.
    example = min(candidates, key=lambda e: (not data.meta["groups"][e["group"]]["template"].startswith("24_"),
                                            abs(e["sigma_requested"]-.1), abs(e["fraction"]-.1)))
    checker = PoseChecker(data, example["index"])
    model = checker.model
    fig = plt.figure(figsize=(16, 8.5), facecolor="#f4f7f9")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.8, 1.], hspace=.24, wspace=.20)
    stages = ("observed", "clean", "tight_clean", "recombined")
    titles = ("Observed pose", "Clean estimate", "Tighter, no residual", "Tighter + original residual")
    colors = ("#526f89", "#207e85", "#207e85", "#526f89")
    reference = checker.inspect(example["poses"]["truth"])[1]
    triangles = checker.surface.triangles
    all_points = np.r_[np.asarray(reference["joints"]), triangles.reshape(-1, 3)]
    center = (all_points.max(0)+all_points.min(0))/2
    radius = np.ptp(all_points, axis=0).max()*.6
    projection_joints = np.asarray(checker.inspect(example["poses"]["projection"])[1]["joints"])
    observed_joints = np.asarray(checker.inspect(example["poses"]["observed"])[1]["joints"])
    for column, (stage, title, color) in enumerate(zip(stages, titles, colors)):
        ax = fig.add_subplot(grid[0, column], projection="3d", facecolor="#f4f7f9")
        _, frame = checker.inspect(example["poses"][stage])
        joints = np.asarray(frame["joints"])
        ax.add_collection3d(Poly3DCollection(triangles, facecolor="#e5b650", edgecolor="none", alpha=.22))
        for finger in range(5):
            chain = joints[[0, 1+4*finger, 2+4*finger, 3+4*finger, 4+4*finger]]
            ax.plot(*chain.T, color=color, linewidth=3.5, marker="o", markersize=3)
        if stage == "clean":
            start, direction = projection_joints[4::4], (observed_joints-projection_joints)[4::4]
            ax.quiver(*start.T, *direction.T, color="#c35a69", arrow_length_ratio=.25, linewidth=1.5)
        for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(center[k]-radius, center[k]+radius)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=45)
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, color="#1c2d3e", pad=6)
        metrics = example["metrics"][stage]
        ax.text2D(.02, .01, f"enclosure {metrics['contact_enclosure_rms_m']*1000:.2f} mm\n"
                  f"object penetration {metrics['hand_object_penetration_m']*1000:.2f} mm",
                  transform=ax.transAxes, fontsize=9, color="#526e87")
    ax = fig.add_subplot(grid[1, :2], facecolor="white")
    positions = np.arange(20)
    for offset, (key, title, color) in enumerate((("known_bias", "Known bias", "#d29a3d"),
                       ("residual", "Extracted residual", "#008a82"),
                       ("noise_perpendicular", "Retained noise", "#b34d5d"))):
        ax.bar(positions+(offset-1)*.25, example[key], width=.25, label=title, color=color)
    ax.set_xlabel("Finger joint"); ax.set_ylabel("Normalized joint value")
    ax.set_title("Adding the residual also adds its noise", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False, ncol=3)
    ax = fig.add_subplot(grid[1, 2:], facecolor="white")
    scores = [s for s in payload["summaries"] if s["split"] == "test"]
    if scores:
        ax.bar([s["method"] for s in scores], [s["macro_rmse_normalized"] for s in scores],
               color=["#008a82" if s["method"] == payload["selected_method"] else "#789bb2" for s in scores])
        ax.set_ylabel("Normalized joint RMSE")
        ax.set_title("Held-out sizes / 3 noise seeds", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.text(.1, .5, "Run evaluate for model comparison", transform=ax.transAxes)
    label = data.meta["groups"][example["group"]]["template"]
    fig.suptitle(f"{label} | {example['method']} | sigma={example['sigma_actual']:.3f} | "
                 f"target size {example['source_size']:.3f} -> {example['target_size']:.3f}",
                 fontsize=15, color="#1c2d3e", y=.97)
    fig.text(.05, .02, "Source object stays fixed. Kinematic candidate poses; no claim of stronger force or stable grasp.",
             fontsize=10, color="#526e87")
    fig.subplots_adjust(left=.05, right=.98, top=.90, bottom=.10)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>抓取流形 · 去噪与收紧实验</title><script>__PLOTLY__</script>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#1c2d3e;font:15px/1.6 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:29px;letter-spacing:-1px;margin:0}p{margin:8px 0 18px}
.eyebrow{font-size:12px;letter-spacing:2px;color:#526e87}.controls,.card{background:white;border:1px solid #dae2e8;border-radius:12px;padding:16px;margin:16px 0}
.controls{display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:5;box-shadow:0 6px 24px #1c2d3e0d}
label{display:flex;flex-direction:column;font-size:12px;color:#526e87;gap:4px}select{font:14px system-ui;padding:7px;border:1px solid #cbd6de;border-radius:6px;background:white;color:#1c2d3e}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.plot{height:420px}.panel{background:white;border:1px solid #dae2e8;border-radius:12px;overflow:hidden}.panel h2{font-size:16px;padding:12px 16px;margin:0;border-bottom:1px solid #eef2f5}
.meta{font-size:13px;color:#526e87}.warn{background:#fff5df;border-left:4px solid #c48728;padding:12px 16px}.metrics{overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px;text-align:right}td,th{padding:9px;border-bottom:1px solid #e5ebef}td:first-child,th:first-child{text-align:left}
.positive{color:#007c69}.negative{color:#b23f38}.legend{display:flex;gap:20px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
#residual{height:260px}.small{font-size:12px;color:#526e87}a{color:#187c9e}@media(max-width:800px){main{padding:12px}.grid{grid-template-columns:1fr}.plot{height:340px}.controls{position:static}}
</style></head><body><main>
<div class="eyebrow">WUJI2 / REPRODUCIBLE MANIFOLD EXPERIMENT</div>
<h1>提取抓取结构，沿尺寸方向收紧</h1>
<p>同一留出抓姿、同一扰动，比较 PCA、尺寸插值、128 原子字典与 diffusion。四个视图中的物体始终保持原尺寸。</p>
<div class="controls">
<label>抓型<select id="group"></select></label><label>模型<select id="method"></select></label>
<label>已知偏置<select id="bias"><option value="none">无偏置</option><option value="orthogonal">正交偏置</option><option value="general">一般方向偏置</option></select></label>
<label>高斯噪声 σ<select id="sigma"></select></label><label>期望尺寸收紧<select id="fraction"></select></label>
<label>叠加结果<select id="clip"><option value="recombined">原始结果</option><option value="clipped">限位裁剪后</option><option value="pca_recombined">纯 PCA 收紧路径</option></select></label>
<label>显示<select id="render"><option value="mesh">手部网格</option><option value="skeleton">骨架与接触点</option></select></label>
</div><div id="status" class="meta"></div>
<div class="legend"><span><i class="dot" style="background:#789bb2"></i>手部</span><span><i class="dot" style="background:#e6b456"></i>固定源物体</span><span><i class="dot" style="background:#008a82"></i>正交残差位移：PCA 投影 → 输入</span></div>
<div class="grid">
<section class="panel"><h2>① 输入抓姿：参考姿态 + 已知偏置 + 噪声</h2><div id="p0" class="plot"></div></section>
<section class="panel"><h2>② 干净姿态估计 / 残差位移</h2><div id="p1" class="plot"></div></section>
<section class="panel"><h2>③ 沿尺寸曲线收紧，暂不加残差</h2><div id="p2" class="plot"></div></section>
<section class="panel"><h2>④ 收紧后叠加原始正交残差</h2><div id="p3" class="plot"></div></section>
</div>
<div class="card metrics"><h2 style="font-size:18px;margin:0 0 10px">收紧改变了什么</h2><div id="metrics"></div></div>
<div class="warn" id="warning"></div>
<div class="card"><h2 style="font-size:18px;margin:0">残差不等于几何偏置</h2><p class="small">关节值按限位归一化。提取残差 = 参考姿态的截断残差 + 偏置的正交部分 + 噪声的正交部分；一般方向偏置的平行部分无法由正交残差保留。</p><div id="residual"></div><div id="resmeta" class="small"></div></div>
<div class="card"><h2 style="font-size:18px;margin:0 0 8px">模型对比</h2><div id="scores"></div><p class="small">默认模型仅根据验证集各类平均误差选择。所有方法已知抓型、primitive 和源尺寸；尺寸插值也拥有同样条件。这里没有真实物体几何偏置标签，也未测量夹持力。</p></div>
<p class="small">完整精度 pose：<a href="poses.npz">poses.npz</a> · 实验与诊断：<a href="comparisons.json">comparisons.json</a> · 拖拽旋转，滚轮缩放；视图相机联动。本页包含全部脚本和网格，可离线打开。</p>
</main><script>
const DATA=__PAYLOAD__;
const $=id=>document.getElementById(id), names={pca:'PCA 正交投影',interpolation:'尺寸插值',dictionary:'128 原子去噪字典',diffusion:'条件 diffusion'};
function options(id,items){for(const [v,label] of items){const o=document.createElement('option');o.value=v;o.textContent=label;$(id).append(o)}}
options('group',DATA.groups.map(g=>[g,DATA.models[g].label]));options('method',DATA.methods.map(m=>[m,names[m]]));
options('sigma',DATA.sigmas.map(s=>[s,s]));options('fraction',DATA.fractions.map(f=>[f,(f*100).toFixed(0)+'%']));
if(DATA.methods.includes(DATA.selected_method))$('method').value=DATA.selected_method;
$('sigma').value=String(DATA.sigmas.includes(.1) ? 0.1 : DATA.sigmas[0]);
$('fraction').value=String(DATA.fractions.includes(.1) ? 0.1 : DATA.fractions[0]);
const add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),scale=(a,s)=>a.map(x=>x*s);
const I=[1,0,0,0,1,0,0,0,1];
function mv(r,v){return [0,1,2].map(i=>r[3*i]*v[0]+r[3*i+1]*v[1]+r[3*i+2]*v[2])}
function mm(a,b){const c=[];for(let i=0;i<3;i++)for(let j=0;j<3;j++)c.push(a[3*i]*b[j]+a[3*i+1]*b[3+j]+a[3*i+2]*b[6+j]);return c}
function qr(q){const [w,x,y,z]=q;return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]}
function rod(a,t){const [x,y,z]=a,c=Math.cos(t),s=Math.sin(t),v=1-c;return [c+x*x*v,x*y*v-z*s,x*z*v+y*s,y*x*v+z*s,c+y*y*v,y*z*v-x*s,z*x*v-y*s,z*y*v+x*s,c+z*z*v]}
function frames(model,q){const poses=[{p:[0,0,0],r:I}];for(let i=1;i<model.bodies.length;i++){const b=model.bodies[i],parent=poses[b.parent];let r=qr(b.quat),p=b.p.slice();for(const j of b.joints){const t=q[j.adr]-j.ref;if(j.type===2)p=add(p,mv(r,scale(j.axis,t)));else{const a=rod(j.axis,t);p=add(p,mv(r,sub(j.pivot,mv(a,j.pivot))));r=mm(r,a)}}poses.push({p:add(parent.p,mv(parent.r,p)),r:mm(parent.r,r)})}return poses}
function point(fr,body,p){return add(fr[body].p,mv(fr[body].r,p))}
function jointPoints(model,fr){const points=[fr[model.chains[0][0]].p];for(let f=0;f<5;f++){points.push(...model.chains[f].slice(1).map(b=>fr[b].p));points.push(point(fr,model.tips[f].body,model.tips[f].p))}return points}
function skeleton(points,color){const x=[],y=[],z=[];for(let f=0;f<5;f++){for(const j of [0,1+4*f,2+4*f,3+4*f,4+4*f]){x.push(points[j][0]);y.push(points[j][1]);z.push(points[j][2])}x.push(null);y.push(null);z.push(null)}return {type:'scatter3d',mode:'lines+markers',x,y,z,line:{color,width:6},marker:{color,size:3},hoverinfo:'skip',showlegend:false}}
function meshTrace(model,fr,isObject){const v=[],f=[];for(const g of model.geoms){if(g.object!==isObject)continue;const src=DATA.meshes[g.mesh],r=mm(fr[g.body].r,qr(g.quat)),p=point(fr,g.body,g.p),offset=v.length;v.push(...src.v.map(x=>add(p,mv(r,x))));f.push(...src.f.map(face=>face.map(i=>i+offset)))}return {type:'mesh3d',x:v.map(x=>x[0]),y:v.map(x=>x[1]),z:v.map(x=>x[2]),i:f.map(x=>x[0]),j:f.map(x=>x[1]),k:f.map(x=>x[2]),color:isObject?'#e6b456':'#789bb2',opacity:isObject?.45:1,flatshading:false,lighting:{ambient:.55,diffuse:.8,specular:.25},hoverinfo:'skip',showlegend:false}}
function traces(model,q,residualTarget,residualSource){const fr=frames(model,q),jp=jointPoints(model,fr),out=[meshTrace(model,fr,true)];if($('render').value==='mesh')out.push(meshTrace(model,fr,false));else out.push(skeleton(jp,'#537c98'));
const pts=model.template.body_names.map((_,i)=>point(fr,model.templateBody[i],model.template.local_points[i]));out.push({type:'scatter3d',mode:'markers',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),marker:{color:'#b34d5d',size:3},hoverinfo:'skip',showlegend:false});
if(residualTarget){const from=jointPoints(model,frames(model,residualSource)),to=jointPoints(model,frames(model,residualTarget)),x=[],y=[],z=[];for(let i=1;i<from.length;i++){x.push(from[i][0],to[i][0],null);y.push(from[i][1],to[i][1],null);z.push(from[i][2],to[i][2],null)}out.push({type:'scatter3d',mode:'lines',x,y,z,line:{color:'#008a82',width:4},hoverinfo:'skip',showlegend:false});const ends=from.map((p,i)=>({p:to[i],d:sub(to[i],p)})).filter(e=>Math.hypot(...e.d)>1e-8);if(ends.length)out.push({type:'cone',x:ends.map(e=>e.p[0]),y:ends.map(e=>e.p[1]),z:ends.map(e=>e.p[2]),u:ends.map(e=>e.d[0]),v:ends.map(e=>e.d[1]),w:ends.map(e=>e.d[2]),anchor:'tip',sizemode:'absolute',sizeref:.003,showscale:false,colorscale:[[0,'#008a82'],[1,'#008a82']],hoverinfo:'skip'})}return out}
// Body names are exported separately from numeric FK arrays for template anchors.
for(const g of DATA.groups){const m=DATA.models[g];m.templateBody=m.template.body_names.map(n=>m.body_names.indexOf(n))}
let camera={eye:{x:1.5,y:1.5,z:1.1},up:{x:0,y:0,z:1}}, syncing=false, plotsReady=false;
const fmt=(x,d=2)=>Number(x).toFixed(d);
async function update(){const group=+$('group').value,method=$('method').value,bias=$('bias').value,sigma=+$('sigma').value,fraction=+$('fraction').value;
const e=DATA.examples.find(e=>e.group===group&&e.method===method&&e.bias===bias&&e.sigma_requested===sigma&&e.fraction===fraction);if(!e)return;
const m=DATA.models[group],last=$('clip').value;
$('status').textContent=`测试节点 ${e.index} · ${m.primitive} / ${e.size_axis} · 源尺寸 ${fmt(e.source_size,4)} → ${fmt(e.target_size,4)} · 实际 σ=${fmt(e.sigma_actual,5)} · 模型种子 ${DATA.model_seeds[method]} / 噪声种子 ${DATA.noise_seed}`;
const stages=['observed','clean','tight_clean',last],truth=jointPoints(m,frames(m,e.poses.truth)),all=[...truth];for(const g of m.geoms.filter(g=>g.object)){const fr=frames(m,e.poses.truth);all.push(...DATA.meshes[g.mesh].v.map(v=>point(fr,g.body,add(g.p,mv(qr(g.quat),v)))))}
const center=[0,1,2].map(i=>(Math.max(...all.map(p=>p[i]))+Math.min(...all.map(p=>p[i])))/2),radius=Math.max(...[0,1,2].map(i=>Math.max(...all.map(p=>p[i]))-Math.min(...all.map(p=>p[i]))))*.65;
for(let i=0;i<4;i++){const scene={aspectmode:'cube',camera,xaxis:{range:[center[0]-radius,center[0]+radius],title:'x (m)',showbackground:false},yaxis:{range:[center[1]-radius,center[1]+radius],title:'y (m)',showbackground:false},zaxis:{range:[center[2]-radius,center[2]+radius],title:'z (m)',showbackground:false}};
await Plotly.react('p'+i,traces(m,e.poses[stages[i]],i===1?e.poses.observed:null,e.poses.projection),{margin:{l:0,r:0,t:0,b:0},scene,paper_bgcolor:'#fff',uirevision:'linked-camera'}, {responsive:true,displayModeBar:false})}
if(!plotsReady){plotsReady=true;for(let i=0;i<4;i++)$('p'+i).on('plotly_relayout',async ev=>{if(syncing||!ev['scene.camera'])return;camera=ev['scene.camera'];syncing=true;await Promise.all([0,1,2,3].filter(j=>j!==i).map(j=>Plotly.relayout('p'+j,{'scene.camera':camera})));syncing=false})}
const selected=['observed','clean','tight_clean',last],labels=['输入','干净估计','收紧无残差','最终叠加'];
const rows=[['接触包络 RMS (mm)','contact_enclosure_rms_m',1000],['活跃指尖平均间距 (mm)','active_fingertip_pair_mean_m',1000],['接触绝对距离 (mm)','contact_absolute_distance_mean_m',1000],['模板锚点覆盖率 (%)','template_point_coverage',100],['最大手物穿透 (mm)','hand_object_penetration_m',1000],['最大自穿透 (mm)','self_penetration_m',1000],['关节越界 (rad)','max_joint_violation_rad',1]];
$('metrics').innerHTML='<table><tr><th>诊断</th>'+labels.map(l=>'<th>'+l+'</th>').join('')+'</tr>'+rows.map(([label,key,k])=>'<tr><td>'+label+'</td>'+selected.map(s=>'<td>'+fmt(e.metrics[s][key]*k, key==='max_joint_violation_rad'?5:2)+'</td>').join('')+'</tr>').join('')+'</table>';
const change=(e.metrics[last].contact_enclosure_rms_m-e.metrics.observed.contact_enclosure_rms_m)*1000;
$('warning').textContent=`期望尺寸减少 ${fmt((1-e.target_size/e.source_size)*100,1)}%，实际接触包络 RMS 变化 ${fmt(change,3)} mm。${change<0?'本例开口指标减小。':'本例该开口指标未减小。'} 加回残差保留的正交噪声 RMS 为 ${fmt(e.retained_noise_rms,5)}。源姿态覆盖率 ${fmt((m.source_quality.coverage||0)*100,1)}%，投影闭合 ${m.source_quality.projected_closure?'通过':'未通过'}；这些数值不代表收紧后稳定抓持。`;
const jointIds=Array.from({length:20},(_,i)=>i+1),bar=(name,y,color)=>({type:'bar',name,x:jointIds,y,marker:{color}});
await Plotly.react('residual',[bar('已知偏置',e.known_bias,'#d29a3d'),bar('提取正交残差',e.residual,'#008a82'),bar('其中：正交噪声',e.noise_perpendicular,'#b34d5d')],{barmode:'group',margin:{l:50,r:15,t:15,b:40},xaxis:{title:'手指关节'},yaxis:{title:'归一化关节值'},legend:{orientation:'h',y:1.15},paper_bgcolor:'#fff',plot_bgcolor:'#fff'},{responsive:true,displayModeBar:false});
$('resmeta').textContent=`分解最大误差 ${e.decomposition_max_error.toExponential(2)}；正交误差 ${e.residual_orthogonality_error.toExponential(2)}；扣除参考截断残差后，与已知偏置的 RMSE ${fmt(e.residual_increment_vs_bias_rmse,5)}；裁剪改变量 ${fmt(e.clipping_pose_change_rad,5)} rad，裁剪导致正交残差改变 ${fmt(e.clipping_residual_change,5)}。`;
}
$('scores').innerHTML=DATA.summaries.length?'<table><tr><th>方法</th><th>验证集 RMSE</th><th>测试集 RMSE</th><th>测试集角度 RMSE (rad)</th></tr>'+DATA.methods.map(method=>{const v=DATA.summaries.find(r=>r.method===method&&r.split==='validation'),t=DATA.summaries.find(r=>r.method===method&&r.split==='test');return '<tr><td>'+names[method]+(method===DATA.selected_method?' · 默认':'')+'</td><td>'+fmt(v.macro_rmse_normalized,5)+'</td><td>'+fmt(t.macro_rmse_normalized,5)+'</td><td>'+fmt(t.mean_rmse_rad,5)+'</td></tr>'}).join('')+'</table>':'尚未运行 evaluate；本页只展示姿态实验。';
for(const id of ['group','method','bias','sigma','fraction','clip','render'])$(id).onchange=()=>update().catch(showError);
function showError(e){$('status').textContent='显示错误：'+e.message;console.error(e)}
update().catch(showError);
</script></body></html>'''
