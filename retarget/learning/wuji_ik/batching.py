"""Exact per-rank frame budgets without truncating any attention context.

A segment crossing a budget boundary is forwarded in full on both sides of
that boundary. Disjoint loss ranges assign each physical row once per epoch.
"""
import math
import random


class FrameBudgetSampler:
    def __init__(self, dataset, frames_per_rank, rank=0, world_size=1, seed=42):
        if frames_per_rank < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid frame budget or rank')
        self.lengths = [s['length'] for _, s in dataset.segments]
        if not self.lengths: raise ValueError('No training segments')
        self.budget = frames_per_rank; self.rank = rank; self.world = world_size; self.seed = seed
        self.optimizer_steps = math.ceil(sum(self.lengths)/(frames_per_rank*world_size))
        self.set_epoch(0)

    def set_epoch(self, epoch):
        order = list(range(len(self.lengths)))
        random.Random(self.seed+epoch).shuffle(order)
        groups = []; all_counts = []; position = 0; cursor = 0
        dummy = min(range(len(self.lengths)), key=self.lengths.__getitem__)
        while position < len(order):
            ranks = []; counts = []
            for rank in range(self.world):
                entries = []; remaining = self.budget
                while remaining and position < len(order):
                    index = order[position]; n = min(remaining, self.lengths[index]-cursor)
                    entries.append((index, cursor, cursor+n)); cursor += n; remaining -= n
                    if cursor == self.lengths[index]: position += 1; cursor = 0
                ranks.append(entries or [(dummy, 0, 0)])
                counts.append(self.budget-remaining)
            groups.append(ranks[self.rank]); all_counts.append(counts)
        self.groups = groups; self.frames_by_rank = all_counts
        if len(groups) != self.optimizer_steps: raise RuntimeError('Frame-budget planner lost rows')

    def __iter__(self):
        for group in self.groups:
            for entry in group: yield [entry]

    def __len__(self):
        return sum(map(len, self.groups))

    def group_at_microbatch(self, consumed):
        cursor = 0
        for index, group in enumerate(self.groups):
            if cursor == consumed: return index
            cursor += len(group)
        if cursor == consumed: return len(self.groups)
        raise ValueError('Resume cursor is not an optimizer-step boundary')
