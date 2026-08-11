"""Toy datasets and graph-construction helpers for the course.

Everything here is generated locally (no downloads) so the notebooks are reproducible
offline and run in seconds.
"""

from __future__ import annotations

import numpy as np
import torch

# Tetris: the classic e3nn toy dataset of 8 four-block 3D shapes.
# Two of them (indices 0 and 1) are mirror images of each other — a *chiral* pair —
# which makes this dataset the standard probe for parity-aware architectures.

TETRIS_LABELS = [
    "chiral_shape_1", "chiral_shape_2", "square", "line",
    "corner", "L", "T", "zigzag",
]


def tetris(dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns positions ``(8, 4, 3)`` and one-hot labels ``(8, 8)``."""
    pos = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
            [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, -1, 0]],  # chiral_shape_2 (mirror)
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
            [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
        ],
        dtype=dtype,
    )
    labels = torch.eye(len(pos), dtype=dtype)
    return pos, labels


# Radius graphs (molecules: no PBC; crystals: PBC via ASE)


def radius_graph(pos: torch.Tensor, r_cut: float, loop: bool = False) -> torch.Tensor:
    """Edge index ``(2, E)`` with an edge (i <- j) whenever |r_j - r_i| < r_cut.

    Brute-force O(N^2): perfectly fine for the small systems used in this course.
    Convention: ``edge_index[0] = receiver i``, ``edge_index[1] = sender j`` — messages
    flow j -> i. This matches the convention stated in STYLE_GUIDE.md.
    """
    dist = torch.cdist(pos, pos)
    adj = dist < r_cut
    if not loop:
        adj.fill_diagonal_(False)
    receiver, sender = adj.nonzero(as_tuple=True)
    return torch.stack([receiver, sender])


def radius_graph_pbc(atoms, r_cut: float):
    """Periodic radius graph from an ASE ``Atoms`` object.

    Returns ``edge_index (2, E)`` and ``edge_shift (E, 3)`` such that the displacement is
    ``r_j + shift @ cell - r_i`` (shift counts periodic images; see Lesson 05a).
    """
    from ase.neighborlist import neighbor_list

    i, j, S = neighbor_list("ijS", atoms, r_cut)
    edge_index = torch.stack([torch.from_numpy(i).long(), torch.from_numpy(j).long()])
    return edge_index, torch.from_numpy(S).to(torch.get_default_dtype())


# Lennard-Jones argon: a tiny energy/forces dataset generated with ASE.
# Used to train the potentials in Parts IV–V without any external download.


def make_lj_argon_dataset(
    n_frames: int = 200,
    n_atoms: int = 8,
    box: float = 7.0,
    temperature_K: float = 300.0,
    seed: int = 0,
    epsilon: float = 0.0104,  # eV  (argon)
    sigma: float = 3.4,       # Å   (argon)
):
    """Short Langevin MD of LJ argon; returns a list of dicts with pos/energy/forces.

    Each frame: ``{"pos": (N,3) float64 tensor, "energy": float, "forces": (N,3) tensor}``.
    Positions are unwrapped (no PBC) to keep early lessons simple: the box only confines
    the initial random placement, and the MD is run without a periodic cell.
    """
    from ase import Atoms, units
    from ase.calculators.lj import LennardJones
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    rng = np.random.default_rng(seed)
    # random initial placement with a minimum-distance rejection loop
    pos = []
    while len(pos) < n_atoms:
        cand = rng.uniform(0, box, 3)
        if all(np.linalg.norm(cand - p) > 0.9 * sigma for p in pos):
            pos.append(cand)
    atoms = Atoms("Ar" * n_atoms, positions=np.array(pos))
    atoms.calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=3.0 * sigma, smooth=True)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    dyn = Langevin(atoms, timestep=2.0 * units.fs, temperature_K=temperature_K,
                   friction=0.02 / units.fs, rng=rng)

    frames = []

    def record():
        frames.append(
            {
                "pos": torch.tensor(atoms.get_positions(), dtype=torch.float64),
                "energy": float(atoms.get_potential_energy()),
                "forces": torch.tensor(atoms.get_forces(), dtype=torch.float64),
            }
        )

    record()
    for _ in range(n_frames - 1):
        dyn.run(5)  # decorrelate frames a bit
        record()
    return frames


def train_val_split(items, val_fraction: float = 0.2, seed: int = 0):
    """Deterministic shuffle-and-split of a list."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = int(round(val_fraction * len(items)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [items[i] for i in train_idx], [items[i] for i in val_idx]
