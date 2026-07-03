"""Plotting helpers shared across the course notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from e3nn import o3


def spherical_surface(n: int = 100):
    """A (theta, phi) grid on the sphere and the corresponding unit vectors ``(n, n, 3)``."""
    theta = np.linspace(0.0, np.pi, n)
    phi = np.linspace(0.0, 2 * np.pi, n)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")
    xyz = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )
    return theta, phi, xyz


def plot_spherical_harmonic(l: int, m: int, ax=None, n: int = 100, cmap: str = "RdBu_r"):
    """Surface plot of Y_l^m: radius = |Y|, color = sign — the classic 'orbital' picture."""
    _, _, xyz = spherical_surface(n)
    pts = torch.from_numpy(xyz.reshape(-1, 3))
    # component m sits at index l + m of the (2l+1)-dim output
    Y = o3.spherical_harmonics(l, pts, normalize=True, normalization="component")
    vals = Y[:, l + m].reshape(xyz.shape[:2]).numpy()
    r = np.abs(vals)
    surf = r[..., None] * xyz

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    vmax = np.abs(vals).max() or 1.0
    colors = plt.get_cmap(cmap)(0.5 * (vals / vmax) + 0.5)
    ax.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], facecolors=colors,
                    rstride=2, cstride=2, linewidth=0.0, antialiased=False)
    lim = max(r.max(), 0.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(rf"$Y_{{{l}}}^{{{m}}}$")
    return ax


def plot_spherical_harmonics_table(l_max: int = 3, n: int = 60):
    """Grid of all Y_l^m for l = 0..l_max (rows), m = -l..l (columns)."""
    fig = plt.figure(figsize=(2.2 * (2 * l_max + 1), 2.2 * (l_max + 1)))
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l * (2 * l_max + 1) + (m + l_max) + 1
            ax = fig.add_subplot(l_max + 1, 2 * l_max + 1, idx, projection="3d")
            plot_spherical_harmonic(l, m, ax=ax, n=n)
    fig.tight_layout()
    return fig


def plot_tp_selection_rules(irreps_1, irreps_2, ax=None):
    """Matrix of output l's allowed by |l1 - l2| <= l3 <= l1 + l2 (with parity product)."""
    irreps_1, irreps_2 = o3.Irreps(irreps_1), o3.Irreps(irreps_2)
    labels_1 = [str(ir) for _, ir in irreps_1]
    labels_2 = [str(ir) for _, ir in irreps_2]
    grid = np.empty((len(labels_1), len(labels_2)), dtype=object)
    for i, (_, ir1) in enumerate(irreps_1):
        for j, (_, ir2) in enumerate(irreps_2):
            grid[i, j] = " + ".join(str(ir) for ir in ir1 * ir2)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 * len(labels_2) + 2, 1.0 * len(labels_1) + 1))
    ax.set_xticks(range(len(labels_2)), labels_2)
    ax.set_yticks(range(len(labels_1)), labels_1)
    ax.set_xlim(-0.5, len(labels_2) - 0.5); ax.set_ylim(len(labels_1) - 0.5, -0.5)
    for i in range(len(labels_1)):
        for j in range(len(labels_2)):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=9)
    ax.set_xlabel("irreps of input 2"); ax.set_ylabel("irreps of input 1")
    ax.set_title(r"$\mathrm{ir}_1 \otimes \mathrm{ir}_2$ (selection rules)")
    ax.grid(True, alpha=0.3)
    return ax


def plot_training_curves(history: dict[str, list[float]], ax=None, logy: bool = True):
    """history: mapping label -> per-epoch values (e.g. {'train E MAE': [...], ...})."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for label, values in history.items():
        ax.plot(values, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss / error")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax


def plot_point_cloud(pos, ax=None, color="C0", label=None, edges=None):
    """3D scatter of atom positions ``(N, 3)``; optionally draw edge segments."""
    pos = np.asarray(pos.detach().cpu() if torch.is_tensor(pos) else pos)
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*pos.T, s=60, c=color, depthshade=True, label=label)
    if edges is not None:
        edges = np.asarray(edges.detach().cpu() if torch.is_tensor(edges) else edges)
        for i, j in edges.T:
            ax.plot(*np.stack([pos[i], pos[j]]).T, c="gray", lw=1, alpha=0.6)
    ax.set_box_aspect((1, 1, 1))
    return ax
