#!/usr/bin/env python3
"""Build the local skill-styled PCA/diffusion noise-and-tightening inspector."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.env.grasp.denoising_inspector import build_inspector
from retarget.env.grasp.manifold import ExperimentData


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=ROOT/'results/grasp_manifold_20261103')
    p.add_argument('--output', type=Path, default=ROOT/'output/grasp_denoising_inspector')
    p.add_argument('--sigma', type=float, default=.05, help='independent Gaussian sigma per round, normalized joint units')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--seeds', default='11,23,37')
    p.add_argument('--fractions', default='0,.1,.2,.35,.5,.65,.8,.9')
    p.add_argument('--device', default='auto')
    p.add_argument('--method', choices=('diffusion', 'diffusion_vq', 'diffusion_vae'), default='diffusion')
    p.add_argument('--noise-mode', choices=('fingers', 'all'), default='fingers')
    p.add_argument('--viser-url', default='http://127.0.0.1:8090')
    a = p.parse_args()
    print(build_inspector(ExperimentData(a.experiment), a.output, sigma=a.sigma,
                          rounds=a.rounds, seeds=tuple(int(v) for v in a.seeds.split(',')),
                          fractions=tuple(float(v) for v in a.fractions.split(',')),
                          device=a.device, viser_url=a.viser_url, method=a.method, noise_mode=a.noise_mode))


if __name__ == '__main__':
    main()
