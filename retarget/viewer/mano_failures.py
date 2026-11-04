"""Portable, read-only diagnostics for saved MANO CEM results."""
from __future__ import annotations

import math
import numpy as np

CATEGORIES = {
    'both_failed': '闭合未通过 · 接触覆盖不足',
    'closure_only_failed': '闭合未通过 · 接触覆盖达标',
    'coverage_only_failed': '闭合通过 · 接触覆盖不足',
    'both_passed': '闭合通过 · 接触覆盖达标',
}


def category(closure, coverage, fraction=.8):
    contact_ok = float(coverage) + 1.e-7 >= fraction
    if closure:
        return 'both_passed' if contact_ok else 'coverage_only_failed'
    return 'closure_only_failed' if contact_ok else 'both_failed'


def slot_diagnostics(distance, hand_normals, object_normals,
                     allowance=.002, tolerance=.002, angle=30.):
    """Same signed gap and normal inequalities as pose_objective.contact_terms."""
    distance = np.asarray(distance)
    cosine = np.clip(-np.sum(hand_normals * object_normals, axis=-1), -1., 1.)
    distance_ok = (distance >= -allowance) & (distance <= tolerance)
    normal_ok = cosine >= math.cos(math.radians(angle))
    return dict(distance_ok=distance_ok, normal_ok=normal_ok,
                fitted=distance_ok & normal_ok,
                normal_angle_deg=np.degrees(np.arccos(cosine)))


def load_case(root, entry):
    with np.load(root / entry['arrays'], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def camera(arrays, reverse=False):
    joints = arrays['joints']
    y = joints[9] - joints[0]
    normal = np.cross(joints[5] - joints[17], y)
    normal /= max(np.linalg.norm(normal), 1.e-9)
    palm = joints[[0, 5, 9, 13, 17]].mean(0)
    if normal @ (arrays['object_vertices'].mean(0) - palm) < 0:
        normal = -normal
    vertices = np.concatenate((arrays['hand_vertices'], arrays['object_vertices']))
    target = vertices.mean(0)
    span = max(np.ptp(vertices, axis=0).max(), .15)
    return target + normal * span * (-2.4 if reverse else 2.4), target, y / max(np.linalg.norm(y), 1.e-9)


def describe(entry, arrays):
    fitted = arrays['fitted']
    distance_ok, normal_ok = arrays['distance_ok'], arrays['normal_ok']
    n = len(fitted)
    mark = lambda value: '通过' if value else '未通过'
    return '\n\n'.join([
        f"**{entry['grasp_type']} / shard {entry['shard']} / row {entry['row']}**",
        f"**投影 QP 闭合：{mark(entry['projected_closure'])}；完整几何验收：{mark(entry['geometry_accepted'])}**",
        f"接触覆盖：**{fitted.sum()}/{n} = {entry['coverage']:.1%}**（要求 ≥80%）",
        f"本例 {n} 个固定槽，至少需 {math.ceil(.8*n-1.e-7)} 个同时合格；3/4=75% 仍未达标。" if n == 4 else f"本例至少需 {math.ceil(.8*n-1.e-7)} 个固定槽同时合格。",
        f"距离合格 {distance_ok.sum()}/{n}；法向合格 {normal_ok.sum()}/{n}；两者同时合格 {fitted.sum()}/{n}。",
        f"仅距离失败 {(~distance_ok & normal_ok).sum()}；仅法向失败 {(distance_ok & ~normal_ok).sum()}；两项都失败 {(~distance_ok & ~normal_ok).sum()}。",
        f"槽间隙范围 {arrays['gap_m'].min()*1000:.2f}～{arrays['gap_m'].max()*1000:.2f} mm；法向误差中位数 {np.median(arrays['normal_angle_deg']):.1f}°。",
        f"手物最大穿透 {entry['object_penetration_m']*1000:.3f} mm；自穿透 {entry['self_penetration_m']*1000:.3f} mm。",
        f"保存的闭合 margin：{entry['closure_margin']:.5g}；闭合 cost：{entry['closure_cost']:.5g}。",
        '闭合判定来自原求解输出的 96 步 QP。未通过表示当前计算未取得证书，不能据此断言严格无解。',
        '接触不足指指定槽位的距离／法向联合覆盖不足，不是单纯数附近顶点；黄色近表面顶点不参与此覆盖分母。',
    ])
