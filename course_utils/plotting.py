"""Plotting helpers shared across the course notebooks.

Two families live here. The ``plot_*`` functions draw static Matplotlib figures. The
``scene3d`` / ``draw_*`` / ``show3d`` trio builds *interactive* Plotly scenes for the 3D
figures -- point clouds, atomistic graphs and spherical harmonics -- which are the ones
where being able to rotate and zoom actually carries meaning.

The interactive figures embed as self-describing HTML, so they stay live both in a local
Jupyter session and on the published book page, with no live kernel behind them.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
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


# Interactive 3D figures with Plotly. Usage mirrors the Matplotlib helpers above, with a
# Plotly figure in place of the axes and an explicit `cell` for multi-panel figures:
#     fig = scene3d(1, 2, titles=["before", "after"])
#     draw_point_cloud(x, fig=fig, cell=(1, 1), color="C0", label="original")
#     draw_point_cloud(y, fig=fig, cell=(1, 2), color="C1", label="rotated")
#     show3d(fig, title="a rotation acts on the whole cloud")


def _as_array(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, dtype=float)


def _css(color):
    """Any Matplotlib colour spec ('C0', 'tab:red', '#abc') -> a CSS hex string."""
    if isinstance(color, (list, tuple, np.ndarray)) and not isinstance(color, str):
        return [_css(c) for c in color]
    return mcolors.to_hex(color)


def scene3d(rows: int = 1, cols: int = 1, titles=None, height: int | None = None,
            spacing: float = 0.04):
    """A Plotly figure of ``rows x cols`` 3D scenes, ready for the ``draw_*`` helpers."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=list(titles) if titles is not None else None,
        horizontal_spacing=spacing, vertical_spacing=spacing,
    )
    fig._course_rows, fig._course_cols = rows, cols
    fig.update_layout(height=height or (360 * rows + 40))
    return fig


def draw_point_cloud(pos, fig=None, cell=(1, 1), color="C0", label=None, edges=None,
                     size: int = 6, edge_color: str = "gray", symbol: str = "circle"):
    """Interactive 3D scatter of positions ``(N, 3)``, optionally with edge segments."""
    import plotly.graph_objects as go

    if fig is None:
        fig = scene3d()
    pos = _as_array(pos)
    row, col = cell

    if edges is not None:
        e = np.asarray(edges.detach().cpu() if torch.is_tensor(edges) else edges)
        # One trace for every segment would be slow; None-separated points give a single
        # trace that Plotly draws as disconnected lines.
        seg = np.full((3 * e.shape[1], 3), np.nan)
        seg[0::3], seg[1::3] = pos[e[0]], pos[e[1]]
        fig.add_trace(
            go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                         line=dict(color=edge_color, width=2), hoverinfo="skip",
                         showlegend=False),
            row=row, col=col,
        )

    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers",
            marker=dict(size=size, color=_css(color), symbol=symbol, line=dict(width=0)),
            name=label or "", showlegend=label is not None,
            hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        ),
        row=row, col=col,
    )
    return fig


def draw_sphere_field(vals, xyz, fig=None, cell=(1, 1), vmax: float | None = None,
                      colorscale: str = "RdBu", stride: int = 1):
    """Any scalar field on the sphere: radius = |f|, surface colour = signed value.

    ``vals`` is ``(n, n)`` (or flat) on the grid returned by :func:`spherical_surface`.
    ``stride`` decimates the grid before it is serialized -- every point ships to the
    browser as JSON, so a fine grid is worth far more in file size than in appearance.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = scene3d()
    vals = _as_array(vals).reshape(xyz.shape[:2])
    if stride > 1:
        vals, xyz = vals[::stride, ::stride], xyz[::stride, ::stride]
    surf = np.abs(vals)[..., None] * xyz
    lim = float(vmax if vmax is not None else (np.abs(vals).max() or 1.0))

    row, col = cell
    fig.add_trace(
        go.Surface(x=surf[..., 0], y=surf[..., 1], z=surf[..., 2],
                   surfacecolor=vals, colorscale=colorscale,
                   cmin=-lim, cmax=lim, showscale=False, hoverinfo="skip"),
        row=row, col=col,
    )
    return fig


def draw_spherical_harmonic(l: int, m: int, fig=None, cell=(1, 1), n: int = 80,
                            colorscale: str = "RdBu"):
    """Interactive ``Y_l^m``: radius = |Y|, surface colour = sign."""
    _, _, xyz = spherical_surface(n)
    Y = o3.spherical_harmonics(l, torch.from_numpy(xyz.reshape(-1, 3)),
                               normalize=True, normalization="component")
    vals = Y[:, l + m].reshape(xyz.shape[:2]).numpy()
    return draw_sphere_field(vals, xyz, fig=fig, cell=cell, colorscale=colorscale)


def show3d(fig, title: str | None = None, axes: bool = False, legend: bool = True,
           aspect: str = "data"):
    """Finish a ``scene3d`` figure and display it as a live, self-contained widget.

    Displays rather than returns, so the figure appears wherever the call sits in the cell
    -- not only when it happens to be the final expression.
    """
    from IPython.display import HTML, display

    n = getattr(fig, "_course_rows", 1) * getattr(fig, "_course_cols", 1)
    axis = dict(visible=axes, showbackground=False)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=48 if title else 16, b=0),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=-0.06, x=0),
        template="plotly_white",
    )
    for i in range(1, n + 1):
        fig.update_layout(**{
            f"scene{i}" if i > 1 else "scene":
                dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode=aspect)
        })
    # include_plotlyjs="cdn" keeps the notebook small while staying self-describing, so the
    # figure is live in Jupyter and on the built page alike.
    # responsive=True makes Plotly size the SVG to its container and re-fit on resize;
    # without it the figure keeps whatever width it was first laid out at and spills out
    # of the output box.
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False,
                             default_width="100%",
                             config={"displaylogo": False, "responsive": True})))
