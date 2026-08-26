# Equivariant Graph Neural Networks with e3nn: A Hands-On Course

This repository offers a comprehensive, step-by-step and user-friendly tutorial
series which takes you from the mathematical foundations of Euclidean symmetry
to a complete and working implementations of state-of-the-art invariant baseline
models **SchNet**, **DimeNet** and equivariant interatomic potentials such as
 **NequIP**, **Allegro**, and **MACE**.

Every lesson follows the same philosophy:

1. **Theory first.** Each operation is derived and motivated mathematically
   *before* presenting any code. We attempt to cite the key manuscripts
   alongside equation numbers therein.
2. **Theory and experiment, side-by-side.** Implementations are broken into code
   blocks of small to moderate sizes while being connected to their
   corresponding equations.
3. **Verifications.** Every key equivariant operation is unit-tested for
   numerically, using shared helper functions from the local `course_utils`
   folder. Feel free to inspect and modify them at your own convenience!
4. **Visualizations.** Spherical harmonics, tensor-product selection rules,
   learned features, training curves, and MD trajectories are plotted throughout
   to assist the reader in building intuition and gaining visual insight.
5. **Modularity and simplicity.** Long lessons are split into `*_a`, `*_b`,
   `*_c` notebooks that can be followed in one sitting, individually.

## Prerequisites

- Linear algebra (matrices, eigenvalues, change of basis), basic group theory
  helps but is introduced from scratch in Lesson 01.
- PyTorch basics (tensors, autograd, `nn.Module`, training loops).
- Some exposure to molecular systems / atomistic simulation is helpful for Parts
  III-V but not required.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd e3nn-course                   # change directory to e3nn-course
uv sync                          # creates .venv with all dependencies
source .venv/bin/activate        # activate the virtual environment
.venv/bin/python -m ipykernel install --user --name e3nn-course   # add the virtual environment to Jupyter kernels (optional)
uv run jupyter lab               # launch Jupyter Lab
```

We heavily rely on the following open-source packages packages: `torch`, `e3nn`,
`torch-geometric`, `ase`, `matplotlib`, and `plotly`.

## Repository Layout

```
e3nn_course/
├── README.md                 <- you are here
├── STYLE_GUIDE.md            <- authoring conventions for all notebooks
├── pyproject.toml / uv.lock  <- reproducible environment (uv)
├── course_utils/             <- shared helpers imported by every notebook
│   ├── equivariance.py       <- numerical equivariance test harness
│   ├── plotting.py           <- spherical harmonics / irreps / training visualizations
│   └── data.py               <- small datasets, neighbor lists, train/val splits
├── notebooks/                <- the lessons (Parts I-V)
└── papers/                   <- primary literature referenced throughout (see below)
```

## Curriculum at a Glance

### Part I: Foundations: Symmetry, Irreps, and Equivariant Operations (Lessons 01-04)

| Notebook | Topic |
|---|---|
| `01a_symmetry_and_equivariance.ipynb` | Why symmetry? Groups, E(3)/SE(3)/O(3), invariance vs. equivariance, why data augmentation is not enough. |
| `01b_group_representations.ipynb` | Representations, reducibility, Schur's lemma in practice, Wigner D-matrices. |
| `02a_irreps_in_e3nn.ipynb` | `e3nn.o3.Irreps`: the type system of equivariant networks - scalars `0e`, vectors `1o`, parity, direct sums. |
| `02b_spherical_harmonics.ipynb` | Spherical harmonics as the equivariant embedding of directions; visualization; `o3.spherical_harmonics`. |
| `03a_tensor_products_theory.ipynb` | Coupling irreps: Clebsch-Gordan coefficients, selection rules, why the tensor product is *the* equivariant bilinear operation. |
| `03b_tensor_products_e3nn.ipynb` | `o3.FullyConnectedTensorProduct` & friends: paths, weights, instructions dissected. |
| `04_nonlinearities_and_gates.ipynb` | Equivariant nonlinearities: norm activations, `e3nn.nn.Gate`; building an equivariant MLP. |

### Part II: From Operations to Networks (Lessons 05-06)

| Notebook | Topic |
|---|---|
| `05a_atomistic_graphs.ipynb` | Point clouds → graphs: cutoffs, neighbor lists, periodic boundary conditions (ASE + torch-geometric). |
| `05b_radial_basis_and_cutoffs.ipynb` | Radial basis functions (Bessel, Gaussian), envelope/cutoff functions, smoothness requirements for potentials. |
| `06a_equivariant_convolution.ipynb` | The equivariant graph convolution (Tensor Field Networks / e3nn point convolution): equations + implementation. |
| `06b_tetris_end_to_end.ipynb` | The classic e3nn "Tetris" exercise: classify chiral 3D shapes with a small equivariant GNN; demonstrate parity. |

### Part III: Invariant Baselines (Lesson 07)

| Notebook | Topic |
|---|---|
| `07a_schnet.ipynb` | SchNet: continuous-filter convolutions; the invariant message-passing blueprint. |
| `07b_dimenet.ipynb` | DimeNet(++): directional message passing with angles; strengths and the incompleteness problem. |
| `07c_rmd17_aspirin.ipynb` | Train SchNet and DimeNet on the small rMD17 aspirin dataset; compare performance. |

### Part IV: State-of-the-Art Equivariant Potentials (Lessons 08-10)

| Notebook | Topic |
|---|---|
| `08a_nequip_theory.ipynb` | NequIP: architecture equations, interaction blocks, why l>0 features boost data efficiency. |
| `08b_nequip_implementation.ipynb` | Block-by-block NequIP in e3nn; equivariance tests; training on a small dataset. |
| `09a_allegro_theory.ipynb` | Allegro: strictly local equivariant descriptors, the scalar track / tensor track design, scalability arguments. |
| `09b_allegro_implementation.ipynb` | Block-by-block Allegro; comparison with NequIP on the same data. |
| `10a_ace_theory.ipynb` | Atomic Cluster Expansion: body order, the density trick, completeness. |
| `10b_mace_theory.ipynb` | MACE: higher-order equivariant message passing = ACE + message passing; the design-space view. |
| `10c_mace_implementation.ipynb` | Block-by-block MACE in e3nn; training and evaluating forces/energies. |

### Part V: Applications (Lesson 11)

| Notebook | Topic |
|---|---|
| `11_molecular_dynamics_ase.ipynb` | Wrap a trained model as an ASE calculator; run MD; sanity checks (energy conservation, RDFs). |

## Primary References

<details> <!-- Start Core frameworks & foundational methods -->
<summary><h3 style="display:inline-block">Core frameworks & foundational methods</h3></summary>

- [Anderson et al., *Cormorant* NeurIPS (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/03573b32b2746e6e8ca98b9123f2249b-Paper.pdf)
- [Battaglia et al., *Relational inductive biases and graph networks* arXiv.1806.01261 (2018)](https://arxiv.org/pdf/1806.01261)
- [Behler & Parrinello, *Generalized neural-network representation of high-dimensional potential-energy surfaces* Phys. Rev. Lett. 98 146401 (2007)](https://doi.org/10.1103/PhysRevLett.98.146401)
- [Fuchs et al., *SE(3)-Transformers* NeurIPS (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)
- [Gasteiger et al., *Directional Message Passing for Molecular Graphs* arXiv.2003.03123 (2020)](https://arxiv.org/pdf/2003.03123)
- [Gasteiger et al., *Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules* arXiv.2011.14115 (2022)](https://arxiv.org/pdf/2011.14115)
- [Geiger & Smidt, *e3nn: Euclidean Neural Networks* arXiv.2207.09453 (2021)](https://arxiv.org/pdf/2207.09453)
- [Liao & Smidt, *Equiformer* ICLR (2023)](https://arxiv.org/pdf/2206.11990)
- [Pozdnyakov & Ceriotti, *Incompleteness of graph neural networks for point clouds in three dimensions* arXiv.2201.07136 2022](https://arxiv.org/pdf/2201.07136)
- [Schütt et al., *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions* arXiv.1706.08566 (2017)](https://arxiv.org/pdf/1706.08566)
- [Schütt et al., *SchNet: A deep learning architecture for molecules and materials*, J. Chem. Phys. 148, 241722 (2018)](https://doi.org/10.1063/1.5019779)
- [Thomas et al., *Tensor Field Networks* arXiv.1802.08219 (2018)](https://arxiv.org/pdf/1802.08219)
- [Weiler et al., *3D Steerable CNNs* NeurIPS (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/488e4104520c6aab692863cc1dba45af-Paper.pdf)

</details> <!-- End Core frameworks & foundational methods -->

<details> <!-- Start Equivariant interatomic potentials -->
<summary><h3 style="display:inline-block">Equivariant interatomic potentials</h3></summary>

- [Batatia et al., *MACE* NeurIPS (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf)
- Batatia et al., *The design space of E(3)-equivariant atom-centred potentials* (2022)
   + [Batatia arXiv.2205.06643 2022](https://arxiv.org/pdf/2205.06643)
   + [Batatia Nat. Mach. Intell. 7 56 2025](https://doi.org/10.1038/s42256-024-00956-x)
- [Batatia et al., *MACE-MP-0 foundation model* arXiv.2401.00096 (2025)](https://arxiv.org/pdf/2401.00096)
- [Batzner et al., *NequIP: E(3)-equivariant GNNs for interatomic potentials* Nat. Commun. 13 2453 (2022)](https://doi.org/10.1038/s41467-022-29939-5)
- [Kovács et al., *Evaluation of MACE* J. Chem. Phys. 159 044118 (2023)](https://doi.org/10.1063/5.0155322)
- [Kovács et al., *MACE-OFF* J. Am. Chem. Soc. 147 17598 (2025)](https://doi.org/10.1021/jacs.4c07099)
- [Musaelian et al., *Allegro: local equivariant representations* Nat. Commun. 14 579 (2023)](https://doi.org/10.1038/s41467-023-36329-y)

</details> <!-- End Equivariant interatomic potentials -->

<details> <!-- Start Atomic Cluster Expansion & theory -->
<summary><h3 style="display:inline-block">Atomic Cluster Expansion & theory</h3></summary>

- [Chong et al., *Resolving the body-order paradox of MLIPs* J. Chem. Phys. 164 064121 (2026)](https://doi.org/10.1063/5.0303302)
- [Drautz, *Atomic cluster expansion* Phys. Rev. B 99 014104 (2019)](https://doi.org/10.1103/PhysRevB.99.014104)
- [Dusson et al., *ACE: completeness, efficiency, stability* J. Comput. Phys. 454 110946 (2022)](https://doi.org/10.1016/j.jcp.2022.110946)
- [Nigam et al., *Unified theory of atom-centered representations and message passing* J. Chem. Phys. 156 204115 (2022)](https://doi.org/10.1063/5.0087042)
- [Pozdnyakov & Ceriotti, *Incompleteness of GNNs for point clouds* arXiv.2201.07136 (2022)](https://arxiv.org/pdf/2201.07136)

</details> <!-- End Atomic Cluster Expansion & theory -->

<details> <!-- Start Long-range interactions & periodic systems -->
<summary><h3 style="display:inline-block">Long-range interactions & periodic systems</h3></summary>

- [Cheng et al., *Latent Ewald summation* npj Comput. Mater. 11 80 (2025)](https://doi.org/10.1038/s41524-025-01577-7)
- [Grisafi & Ceriotti, *Incorporating long-range physics* J. Chem. Phys. 151 204105 (2019)](https://doi.org/10.1063/1.5128375)
- [Kolafa & Perram, *Cutoff errors in Ewald summation* Mol. Simul. 9 351 (1992)](https://doi.org/10.1080/08927029208049126)
- [Kosmala et al., *Ewald-based message passing*  PMLR 202 17544 (2023)](https://proceedings.mlr.press/v202/kosmala23a.html)

</details> <!-- End Long-range interactions & periodic systems -->

<details> <!-- Start Additional resources -->
<summary><h3 style="display:inline-block">Additional resources</h3></summary>

- [The official e3nn MRS Fall 2021 tutorial](https://e3nn.org/e3nn-tutorial-mrs-fall-2021)

</details> <!-- End Additional resources -->