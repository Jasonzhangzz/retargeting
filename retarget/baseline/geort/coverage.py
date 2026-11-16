"""Incremental coverage of saved fingertip samples, without torch or Open3D.

Counts use a fixed wrist-coordinate voxel grid. Reachable-space percentages
are computed separately against the calibrated FK reference.
"""
from collections import deque, Counter
import copy
import time
import numpy as np
from retarget.baseline.geort.adapter import FINGERS

VOXEL_M = .001
DISPLAY_VOXEL_M = .005


class WorkspaceCoverage:
    def __init__(self, source):
        self.source = source
        self.frames = 0
        self.voxels = [set() for _ in FINGERS]
        self.coarse = [set() for _ in FINGERS]
        self.coarse_counts = [Counter() for _ in FINGERS]
        self.lower = np.full((5, 3), np.inf)
        self.upper = np.full((5, 3), -np.inf)
        self.recent = deque()
        self.last_stamp = None
        self.pinches = [dict(frames=0, seconds=0., episodes=0, positions=set(),
                             active=False, elapsed=0., counted=False) for _ in range(4)]

    def clone(self):
        result = copy.deepcopy(self)
        result.reset_time()
        result.recent.clear()
        return result

    def reset_time(self):
        self.last_stamp = None
        for p in self.pinches:
            p.update(active=False, elapsed=0., counted=False)

    def add(self, points, stamp, now=None):
        points = np.asarray(points, dtype=float)
        if points.shape != (5, 3) or not np.isfinite(points).all() or not np.isfinite(stamp):
            raise ValueError('覆盖评估需要五指有限坐标和时间戳')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        dt = (stamp-self.last_stamp)/1000 if self.last_stamp is not None else 0.
        if dt > .25:
            self.reset_time()
            dt = 0.
        self.last_stamp = stamp
        fine = np.floor(points/VOXEL_M + 1e-7).astype(np.int64)
        coarse = np.floor(points/DISPLAY_VOXEL_M + 1e-7).astype(np.int64)
        added = []
        for i in range(5):
            key = tuple(fine[i])
            added.append(int(key not in self.voxels[i]))
            self.voxels[i].add(key)
            self.coarse[i].add(tuple(coarse[i]))
            self.coarse_counts[i][tuple(coarse[i])]+=1
        self.lower = np.minimum(self.lower, points)
        self.upper = np.maximum(self.upper, points)
        self.frames += 1
        now = time.monotonic() if now is None else now
        self.recent.append((now, added))
        self._expire(now)
        for i, p in enumerate(self.pinches):
            distance = np.linalg.norm(points[0]-points[i+1])
            if p['active'] and distance >= .020:
                p.update(active=False, elapsed=0., counted=False)
            if distance < .015:
                continuing = p['active']
                p['active'] = True
                p['frames'] += 1
                p['positions'].add(tuple(np.floor((points[0]+points[i+1])/(2*VOXEL_M)+1e-7).astype(np.int64)))
                p['seconds'] += dt if continuing else 0.
                p['elapsed'] += dt if continuing else 0.
                if p['elapsed'] >= .1 and not p['counted']:
                    p['episodes'] += 1
                    p['counted'] = True
        return True

    def _expire(self, now):
        while self.recent and self.recent[0][0] < now-5:
            self.recent.popleft()

    def summary(self, include_cloud=False, now=None):
        self._expire(time.monotonic() if now is None else now)
        growth = np.sum([row[1] for row in self.recent], axis=0) if self.recent else np.zeros(5)
        fingers = []
        for i, name in enumerate(FINGERS):
            item = dict(finger=name, unique_voxels=len(self.voxels[i]), coarse_voxels=len(self.coarse[i]),
                        added_last_5s=int(growth[i]),
                        span_mm=np.round((self.upper[i]-self.lower[i])*1000, 1).tolist() if self.frames else [0., 0., 0.])
            if include_cloud:
                cells = sorted(self.voxels[i])
                # Deterministic bounded rendering; statistics always count all cells.
                cells = cells[::max(1, (len(cells)+1999)//2000)]
                item['cloud_kind']='recorded_1mm_voxel_centres'
                item['cloud_mm'] = ((np.asarray(cells).reshape(-1, 3)+.5)*VOXEL_M*1000).round(2).tolist()
            fingers.append(item)
        pinches = [dict(finger=FINGERS[i+1], frames=p['frames'], seconds=round(p['seconds'], 2),
                        episodes=p['episodes'], unique_positions=len(p['positions'])) for i, p in enumerate(self.pinches)]
        return dict(frames=self.frames, voxel_mm=1, display_voxel_mm=5,
                    fingers=fingers, pinches=pinches, pinch_threshold_mm=15,
                    grid='fixed_wrist_origin', absolute_workspace_percentage=False)
