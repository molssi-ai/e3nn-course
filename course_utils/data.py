"""Toy datasets and graph-construction helpers for the course.

Everything here is generated locally so the notebooks are reproducible offline and run
in seconds. The one exception is ``load_rmd17`` (Lesson 07c), which downloads a real
molecular dataset on first use and then works from a small local cache.
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


# Revised MD17 (rMD17): real DFT energies/forces for ten small organic molecules.
# Christensen & von Lilienfeld, "On the role of gradients for machine learning of
# molecular energies and forces" (Mach. Learn.: Sci. Technol. 2020); data on figshare
# (doi:10.6084/m9.figshare.12672038). PBE/def2-SVP with very tight SCF and dense grids,
# so the forces are practically noise-free. Used in Lesson 07c.

_RMD17_FIGSHARE = "https://ndownloader.figshare.com/files/{}"

_RMD17_MOLECULES = {  # molecule -> figshare file id of rmd17_<molecule>.npz
    "aspirin": 62265757, "azobenzene": 62265754, "benzene": 62265739,
    "ethanol": 62265733, "malonaldehyde": 62265736, "naphthalene": 62265751,
    "paracetamol": 62265760, "salicylic": 62265748, "toluene": 62265742,
    "uracil": 62265745,
}

_RMD17_SPLIT_01 = {"train": 62265793, "test": 62265781}  # official split 01 index CSVs

KCAL_PER_MOL = 0.0433641  # eV; rMD17 stores energies/forces in kcal/mol (and Å)


def _download(url: str, dest, desc: str):
    """Tiny urllib downloader (figshare rejects requests without a browser user agent)."""
    import urllib.request

    print(f"downloading {desc} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def load_rmd17(molecule: str = "aspirin", n_train: int = 150, n_val: int = 50,
               cache_dir: str = "artifacts"):
    """A small, deterministic subset of the revised MD17 dataset.

    On first call this downloads the full per-molecule file (70-180 MB) plus the
    official train/test index lists of split 01, keeps the 1000 train + 1000 test
    frames of that split in a small cached ``.npz`` (~2 MB) under ``cache_dir``, and
    deletes the big download. Every later call (any ``n_train``/``n_val``) is offline.

    Returns ``(train_frames, val_frames, z)``: frames follow the course convention
    ``{"pos": (N,3) float64 [Å], "energy": float [eV], "forces": (N,3) [eV/Å]}``
    (converted from the dataset's kcal/mol), and ``z`` is the ``(N,)`` tensor of
    nuclear charges, identical for every frame. ``train_frames`` are the first
    ``n_train`` frames of the official train split, ``val_frames`` the first
    ``n_val`` of the official test split — no seed involved.

    The dataset readme warns that frames come from a MD trajectory (correlated
    samples): never train on more than the 1000 frames of the official split.
    """
    import os

    if molecule not in _RMD17_MOLECULES:
        raise ValueError(f"unknown molecule {molecule!r}; one of {sorted(_RMD17_MOLECULES)}")
    if not (0 < n_train <= 1000 and 0 < n_val <= 1000):
        raise ValueError("n_train and n_val must be in [1, 1000] (correlated MD frames!)")

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"rmd17_{molecule}_split01.npz")

    if not os.path.exists(cache):
        # local fallback: a manually placed full download is picked up instead
        raw = os.path.join(cache_dir, f"rmd17_{molecule}_full.npz")
        if not os.path.exists(raw):
            _download(_RMD17_FIGSHARE.format(_RMD17_MOLECULES[molecule]), raw,
                      f"rmd17_{molecule}.npz (one-time, ~100 MB)")
        idx = {}
        for part, file_id in _RMD17_SPLIT_01.items():
            index_csv = os.path.join(cache_dir, f"rmd17_index_{part}_01.csv")
            if not os.path.exists(index_csv):
                _download(_RMD17_FIGSHARE.format(file_id), index_csv,
                          f"split-01 {part} indices")
            idx[part] = np.loadtxt(index_csv, dtype=int)
        full = np.load(raw)
        np.savez_compressed(
            cache, nuclear_charges=full["nuclear_charges"],
            **{f"{k}_{p}": full[k][idx[p]] for p in ("train", "test")
               for k in ("coords", "energies", "forces")},
            **{f"index_{p}": idx[p] for p in ("train", "test")},
        )
        os.remove(raw)  # keep only the ~2 MB subset
        print(f"cached {cache}")

    data = np.load(cache)
    z = torch.tensor(data["nuclear_charges"].astype(np.int64))

    def frames(part, n):
        return [
            {"pos": torch.tensor(data[f"coords_{part}"][i]),
             "energy": float(data[f"energies_{part}"][i]) * KCAL_PER_MOL,
             "forces": torch.tensor(data[f"forces_{part}"][i]) * KCAL_PER_MOL}
            for i in range(n)
        ]

    return frames("train", n_train), frames("test", n_val), z


def train_val_split(items, val_fraction: float = 0.2, seed: int = 0):
    """Deterministic shuffle-and-split of a list."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = int(round(val_fraction * len(items)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [items[i] for i in train_idx], [items[i] for i in val_idx]
