"""Export metric learning curves and a moving link's native/learned SDF slices."""
import json
from pathlib import Path

import numpy as np
import torch


def plot(model_path, output, device='cpu'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .geometry import Geometry
    from .model import load_model
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, saved = load_model(model_path, device)
    teacher = Geometry(device)
    if saved['geometry']['asset_sha256'] != teacher.surface.asset_hash:
        raise ValueError('Native geometry changed since training')
    # One fixed plane intersects the native distal surface across these poses.
    y = np.linspace(-.035, .11, 180)
    z = np.linspace(-.195, -.055, 180)
    yy, zz = np.meshgrid(y, z)
    points = np.stack((np.full_like(yy, .032), yy, zz), -1).reshape(-1, 3).astype(np.float32)
    q = np.zeros((3, 20), np.float32)
    q[:, [4, 6, 7]] = [[.15, .2, .2], [.45, .6, .5], [.75, 1., .8]]
    xyz = np.broadcast_to(points[None], (3, len(points), 3)).copy()
    links = np.full((3, len(points)), 8, np.int32)
    actual = teacher.label(xyz, links, teacher.frames(q)).reshape(3, len(z), len(y))
    with torch.no_grad():
        predictions = []
        for i in range(3):
            predictions.append(torch.cat([model(torch.as_tensor(q[i], device=device),
                torch.as_tensor(p, device=device), torch.full((len(p),), 8, device=device))
                for p in np.array_split(points, 8)]).cpu().numpy().reshape(len(z), len(y)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for row, fields in enumerate((actual, predictions)):
        for column, field in enumerate(fields):
            ax = axes[row, column]
            artist = ax.pcolormesh(y * 1000, z * 1000, np.clip(field * 1000, -20, 20),
                                   cmap='RdBu', vmin=-20, vmax=20, shading='auto', rasterized=True)
            if field.min() <= 0 <= field.max():
                ax.contour(y * 1000, z * 1000, field, levels=[0], colors='black', linewidths=1.)
            ax.contour(y * 1000, z * 1000, actual[column], levels=[0], colors='limegreen',
                       linewidths=.8, linestyles='--')
            ax.set_aspect('equal')
            ax.set_title(('Native mesh' if row == 0 else 'MLP (x, q, link)') + f' | pose {column + 1}')
            if row == 1:
                ax.set_xlabel('wrist y (mm)')
            if column == 0:
                ax.set_ylabel('wrist z (mm)')
    fig.colorbar(artist, ax=axes, label='signed distance (mm), negative inside', shrink=.75)
    fig.suptitle('Index distal link, fixed x = 32 mm; green dashed = native zero level')
    fig.savefig(output / 'configuration_slices.png', dpi=160)
    plt.close(fig)
    history_path = Path(model_path).parent / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        epochs = [r['epoch'] for r in history]
        for metric in ('mae_mm', 'near_mae_mm'):
            axes[0].plot(epochs, [r['validation'][metric] for r in history], label=metric)
        axes[0].set(xlabel='epoch', ylabel='validation distance MAE (mm)')
        axes[0].legend()
        axes[1].plot(epochs, [r['validation']['penetration_false_negative_rate'] for r in history])
        axes[1].set(xlabel='epoch', ylabel='validation penetration false negative rate', ylim=(0, 1))
        fig.savefig(output / 'training_curves.png', dpi=160)
        plt.close(fig)
    return str(output / 'configuration_slices.png')
