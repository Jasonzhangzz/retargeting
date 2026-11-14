"""Input-only continuous EMF screening and deterministic frame allocation."""
import numpy as np


def continuous_emf_ranges(position, quaternion, confidence, timestamp_ns, source_seq,
                          extra_valid=None, min_frames=64, max_frames=1024, expected_seq_step=None):
    p, q, c = map(np.asarray, (position, quaternion, confidence))
    ts, seq = np.asarray(timestamp_ns), np.asarray(source_seq)
    n = len(p)
    if p.shape != (n, 5, 3) or q.shape != (n, 5, 4) or c.shape != (n, 5):
        raise ValueError('Expected five recorded EMF poses per frame')
    if ts.shape != (n,) or seq.shape != (n,) or min_frames < 1 or max_frames < min_frames:
        raise ValueError('Invalid frame metadata or segment bounds')
    norm = np.linalg.norm(q, axis=-1)
    distance = np.linalg.norm(p, axis=-1)
    gates = {
        'finite': np.isfinite(p).all((1, 2)) & np.isfinite(q).all((1, 2)) & np.isfinite(c).all(1),
        'confidence': (c >= .8).all(1) & (c <= 1.00001).all(1),
        'radius': (distance >= .01).all(1) & (distance <= .25).all(1),
        'quaternion': (abs(norm - 1) <= .002).all(1),
    }
    if extra_valid is not None:
        gates['additional_input_validity'] = np.asarray(extra_valid, dtype=bool)
        if gates['additional_input_validity'].shape != (n,):
            raise ValueError('Invalid kinematic gate')
    valid = np.logical_and.reduce(list(gates.values()))
    if n < 2:
        return [], dict(frames=n, valid_frames=int(valid.sum()), kept_frames=0,
                        rejected_by_gate={k: int((~v).sum()) for k, v in gates.items()})
    dt = np.diff(ts).astype(np.float64) * 1e-9
    ds = np.diff(seq)
    positive = ds[ds > 0]
    nominal_step = int(expected_seq_step) if expected_seq_step is not None else (int(np.median(positive)) if len(positive) else 1)
    qn = q / np.maximum(norm[..., None], 1e-12)
    rotation_step = 2*np.arccos(np.clip(abs((qn[1:]*qn[:-1]).sum(-1)), 0, 1))
    translation_step = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    edges = ((dt > 0) & (dt <= .05) & (ds == nominal_step)
             & (translation_step <= .025).all(1) & (rotation_step <= .5).all(1)
             & valid[:-1] & valid[1:])
    ranges = []
    start = None
    for i in range(n):
        if valid[i] and (i == 0 or not edges[i-1]):
            start = i
        if valid[i] and (i == n-1 or not edges[i]):
            end = i+1
            if end-start >= min_frames:
                # Balanced contiguous chunks avoid a tiny discarded tail.
                parts = int(np.ceil((end-start)/max_frames))
                cuts = np.linspace(start, end, parts+1, dtype=np.int64)
                ranges.extend((int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]))
            start = None
    return ranges, dict(frames=n, valid_frames=int(valid.sum()), kept_frames=sum(b-a for a, b in ranges),
                        nominal_source_seq_step=nominal_step, broken_edges=int((~edges).sum()),
                        rejected_by_gate={k: int((~v).sum()) for k, v in gates.items()})


def allocate_even(capacities, total):
    """Exact, capacity-limited equal allocation; zero-capacity groups stay zero."""
    if total < 0 or any(v < 0 for v in capacities.values()) or sum(capacities.values()) < total:
        raise ValueError('Insufficient nonnegative capacity for requested allocation')
    result = {k: 0 for k in capacities}
    active = sorted(k for k, v in capacities.items() if v)
    left = total
    while active and left:
        base, remainder = divmod(left, len(active))
        small = [k for i, k in enumerate(active) if capacities[k] <= base+(i < remainder)]
        if small:
            for k in small:
                result[k] = capacities[k]
                left -= capacities[k]
            active = [k for k in active if k not in small]
        else:
            for i, k in enumerate(active):
                result[k] = base+int(i < remainder)
            left = 0
    if sum(result.values()) != total or any(result[k] > capacities[k] for k in result):
        raise AssertionError('Frame allocator lost or duplicated frames')
    return result


def diverse_windows(candidates, budget, rng, prefix_only=False):
    """Cycle motion/spread bins; select disjoint contiguous subsegments only."""
    if budget == 0:
        return []
    if sum(x['end']-x['start'] for x in candidates) < budget:
        raise ValueError('Insufficient unique continuous frames')
    values = np.array([[x['activity'], x['spread']] for x in candidates])
    cuts = np.quantile(values, [1/3, 2/3], axis=0)
    bins = (values > cuts[0]).astype(int)+(values > cuts[1]).astype(int)
    buckets = {}
    for i, cell in enumerate(bins):
        buckets.setdefault(tuple(cell), []).append(i)
    for cell in buckets:
        rng.shuffle(buckets[cell])
    order = []
    while any(buckets.values()):
        # Collection quotas often fit inside a single window. Always starting
        # at (0, 0) would therefore select mostly low-motion closed poses.
        # Randomize activity first, independently of the number of spread bins
        # available at each activity level; then sample an available spread bin.
        activity_bins = sorted({cell[0] for cell, ids in buckets.items() if ids})
        rng.shuffle(activity_bins)
        for activity in activity_bins:
            cells = sorted(cell for cell, ids in buckets.items() if ids and cell[0] == activity)
            cell = cells[int(rng.integers(len(cells)))]
            order.append(buckets[cell].pop())
    result = []
    for i in order:
        if not budget:
            break
        candidate = dict(candidates[i])
        length = min(budget, candidate['end']-candidate['start'])
        # The SDK and network both start fresh at this selected segment's start.
        start = (candidate['start'] if prefix_only else
                 int(rng.integers(candidate['start'], candidate['end']-length+1)))
        candidate.update(start=start, end=start+length, motion_bin=bins[i].tolist())
        result.append(candidate)
        budget -= length
    return result
