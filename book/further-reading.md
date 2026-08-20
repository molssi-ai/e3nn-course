# Further reading

The primary literature behind the course, grouped by theme, with the lesson that uses each
work. All links are DOIs or arXiv entries.

## Core frameworks and foundational methods

| Work | Used in |
|---|---|
| [Geiger & Smidt, *e3nn: Euclidean Neural Networks*, arXiv:2207.09453 (2022)](https://arxiv.org/abs/2207.09453) | throughout |
| [Thomas et al., *Tensor Field Networks*, arXiv:1802.08219 (2018)](https://arxiv.org/abs/1802.08219) | 03a, 06a |
| [Weiler et al., *3D Steerable CNNs*, NeurIPS (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/488e4104520c6aab692863cc1dba45af-Paper.pdf) | 01b, 06a |
| [Battaglia et al., *Relational inductive biases and graph networks*, arXiv:1806.01261 (2018)](https://arxiv.org/abs/1806.01261) | 05a |
| [Fuchs et al., *SE(3)-Transformers*, NeurIPS (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf) | 06a |
| [Anderson et al., *Cormorant*, NeurIPS (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/file/03573b32b2746e6e8ca98b9123f2249b-Paper.pdf) | 03a |
| [Liao & Smidt, *Equiformer*, ICLR (2023)](https://arxiv.org/abs/2206.11990) | 04 |

## Invariant baselines

| Work | Used in |
|---|---|
| [Behler & Parrinello, *Generalized neural-network representation of high-dimensional potential-energy surfaces*, Phys. Rev. Lett. **98** 146401 (2007)](https://doi.org/10.1103/PhysRevLett.98.146401) | 05b, 08a |
| [Schütt et al., *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*, arXiv:1706.08566 (2017)](https://arxiv.org/abs/1706.08566) | 07a |
| [Schütt et al., *SchNet — A deep learning architecture for molecules and materials*, J. Chem. Phys. **148** 241722 (2018)](https://doi.org/10.1063/1.5019779) | 07a |
| [Gasteiger et al., *Directional message passing for molecular graphs* (DimeNet), arXiv:2003.03123 (2022)](https://arxiv.org/abs/2003.03123) | 07b |
| [Gasteiger et al., *DimeNet++*, arXiv:2011.14115 (2020)](https://arxiv.org/abs/2011.14115) | 07b |

## Equivariant interatomic potentials

| Work | Used in |
|---|---|
| [Batzner et al., *NequIP*, Nat. Commun. **13** 2453 (2022)](https://doi.org/10.1038/s41467-022-29939-5) | 08a, 08b |
| [Musaelian et al., *Allegro*, Nat. Commun. **14** 579 (2023)](https://doi.org/10.1038/s41467-023-36329-y) | 09a, 09b |
| [Batatia et al., *MACE*, NeurIPS (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/4a36c3c51af11ed9f34615b81edb5bbc-Paper-Conference.pdf) | 10b, 10c |
| [Batatia et al., *The design space of E(3)-equivariant potentials*, arXiv:2205.06643 (2022)](https://arxiv.org/abs/2205.06643) · [Nat. Mach. Intell. **7** 56 (2025)](https://doi.org/10.1038/s42256-024-00956-x) | 10b |
| [Kovács et al., *Evaluation of MACE*, J. Chem. Phys. **159** 044118 (2023)](https://doi.org/10.1063/5.0155322) | 10c |
| [Kovács et al., *MACE-OFF*, J. Am. Chem. Soc. **147** 17598 (2025)](https://doi.org/10.1021/jacs.4c07099) | 10c |
| [Batatia et al., *MACE-MP-0 foundation model*, arXiv:2401.00096 (2024)](https://arxiv.org/abs/2401.00096) | 11 |

## Atomic Cluster Expansion and representation theory

| Work | Used in |
|---|---|
| [Drautz, *Atomic cluster expansion*, Phys. Rev. B **99** 014104 (2019)](https://doi.org/10.1103/PhysRevB.99.014104) | 10a |
| [Dusson et al., *ACE: completeness, efficiency, stability*, J. Comput. Phys. **454** 110946 (2022)](https://doi.org/10.1016/j.jcp.2022.110946) | 10a |
| [Nigam et al., *Unified theory of atom-centered representations and message passing*, J. Chem. Phys. **156** 204115 (2022)](https://doi.org/10.1063/5.0087042) | 10a, 10b |
| [Pozdnyakov & Ceriotti, *Incompleteness of GNNs for point clouds*, arXiv:2201.07136 (2022)](https://arxiv.org/abs/2201.07136) | 07b, 10a |
| [Chong et al., *Resolving the body-order paradox of MLIPs*, J. Chem. Phys. **164** 064121 (2026)](https://doi.org/10.1063/5.0303302) | 10a |

## Long-range interactions and periodic systems

| Work | Used in |
|---|---|
| [Grisafi & Ceriotti, *Incorporating long-range physics*, J. Chem. Phys. **151** 204105 (2019)](https://doi.org/10.1063/1.5128375) | 05a |
| [Kosmala et al., *Ewald-based message passing*, PMLR **202** 17544 (2023)](https://proceedings.mlr.press/v202/kosmala23a.html) | 05a |
| [Cheng, *Latent Ewald summation*, npj Comput. Mater. **11** 80 (2025)](https://doi.org/10.1038/s41524-025-01577-7) | 05a |
| [Kolafa & Perram, *Cutoff errors in Ewald summation*, Mol. Simul. **9** 351 (1992)](https://doi.org/10.1080/08927029208049126) | 05b |

## Software documentation

- [e3nn documentation](https://docs.e3nn.org) — the API this course teaches
- [The official e3nn MRS Fall 2021 tutorial](https://e3nn.org/e3nn-tutorial-mrs-fall-2021) — a
  shorter, complementary walkthrough by the library authors
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io) — graph
  batching and scatter operations
- [ASE documentation](https://wiki.fysik.dtu.dk/ase/) — structures, calculators, MD
- [MACE](https://github.com/ACEsuit/mace) · [NequIP](https://github.com/mir-group/nequip) ·
  [Allegro](https://github.com/mir-group/allegro) — the reference implementations of the
  potentials rebuilt in Part IV
