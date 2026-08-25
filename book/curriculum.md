# Curriculum at a glance

Twenty-two lessons in five parts. Theory lessons are lettered `a`, implementations `b`/`c`,
and each is self-contained enough to open cold.

## Part I: Foundations: Symmetry, Irreps, and Equivariant Operations (Lessons 01-04)

*From "why symmetry at all" to a working equivariant MLP that generalizes to unseen
orientations with zero augmentation.*

| Lesson | What you learn | Key API |
|---|---|---|
| [01a · Symmetry and equivariance](../notebooks/01a_symmetry_and_equivariance.ipynb) | $\mathrm{E}(3)$ and its subgroups $\mathrm{SO}(3) \subset \mathrm{O}(3)$; the precise difference between **invariance** (energies) and **equivariance** (forces, dipoles); a numerical demonstration that data augmentation only ever *approximates* a symmetry it never enforces | — |
| [01b · Group representations](../notebooks/01b_group_representations.ipynb) | Why the homomorphism $D(g_1 g_2) = D(g_1)D(g_2)$ is exactly what a layer needs to know about its data; reducible vs. irreducible; the irreps of $\mathrm{SO}(3)$ labelled by $l$ with dimension $2l+1$; Schur orthogonality; parity | `o3.wigner_D` |
| [02a · Irreps as a type system](../notebooks/02a_irreps_in_e3nn.ipynb) | Reading `"16x0e + 8x1o"` — multiplicity, degree, parity; how e3nn lays features out in memory; which operations preserve equivariance and which quietly break it; **pseudo**-scalars and -vectors via chirality and angular momentum | `o3.Irreps`, `Irreps.D_from_matrix` |
| [02b · Spherical harmonics](../notebooks/02b_spherical_harmonics.ipynb) | The equivariant embedding of a *direction*, $Y^{(l)}_{}(R\hat{\mathbf{r}}) = D^{(l)}(R)\,Y^{(l)}_{}(\hat{\mathbf{r}})$; parity $(-1)^l$; the three normalization conventions and when each matters; expanding a function on the sphere, and why truncating at $L$ is a low-pass filter | `o3.spherical_harmonics` |
| [03a · Tensor products (theory)](../notebooks/03a_tensor_products_theory.ipynb) | Clebsch–Gordan decomposition and the parity rule $p_3 = p_1 p_2$; that the dot product, cross product and symmetric traceless outer product are *exactly* the three paths of $1 \otimes 1$; why Schur's lemma makes the tensor product the **only** equivariant bilinear map | — |
| [03b · Tensor products in e3nn](../notebooks/03b_tensor_products_e3nn.ipynb) | Turning that decomposition into a **learnable layer**, $x \otimes_{w} y$; instructions and connection modes (`uvw`, `uvu`, …); reproducing a tensor product with a bare `einsum`; counting weights; why a plain `nn.Linear` on raw components breaks equivariance | `o3.TensorProduct`, `o3.FullyConnectedTensorProduct` |
| [04 · Nonlinearities and gates](../notebooks/04_nonlinearities_and_gates.ipynb) | Why pointwise `tanh`/`ReLU` destroys equivariance for $l>0$ — argued, then measured; the two escape hatches (functions of invariants, norm activations); the gated nonlinearity dissected and rebuilt by hand; a parity trap where an even activation on a `0o` scalar passes every $\mathrm{SO}(3)$ test and still breaks $\mathrm{O}(3)$ | `e3nn.nn.Gate`, `NormActivation` |

## Part II: From Operations to Networks (Lessons 05-06)

*The geometry and bookkeeping every model in Parts III–V depends on, then the convolution
the rest of the course is variations on.*

| Lesson | What you learn | Key API |
|---|---|---|
| [05a · Atomistic graphs](../notebooks/05a_atomistic_graphs.ipynb) | Molecules and crystals as radius graphs; the graph-network message-passing framework; the course-wide edge convention $\vec{r}_{ij} = \vec{r}_{j} - \vec{r}_{i}$ and why absolute positions never enter a network; periodic boundary conditions and the extra `edge_shift` periodic edges require | `ase.neighborlist`, `radius_graph` |
| [05b · Radial bases and cutoffs](../notebooks/05b_radial_basis_and_cutoffs.ipynb) | Why a raw distance is a poor input; Gaussian (SchNet) vs. Bessel (DimeNet) bases; why the energy must be at least $C^1$ at $r_\mathrm{cut}$ for forces to exist — plus a numerical demonstration of the force artifacts a hard cutoff produces | cosine cutoff, polynomial envelope |
| [06a · The equivariant convolution](../notebooks/06a_equivariant_convolution.ipynb) | **The core lesson.** The Tensor Field Networks point convolution, $m_{ij} = (R(\lvert\vec{r}_{ij}\rvert)\,Y(\hat r_{ij})) \otimes_{w} h_j$; a factor-by-factor equivariance proof; a step-by-step build; how the receptive field grows with depth | full TFN block in e3nn |
| [06b · Tetris, end to end](../notebooks/06b_tetris_end_to_end.ipynb) | Batching many small graphs; why a model whose output is only true scalars (`0e`) **provably cannot** separate mirror images — verified before *and* after training; why a pseudoscalar (`0o`) fixes it and needs three tensor-product layers; training on one orientation and scoring 100% on random rotations | scatter batching |

## Part III: Invariant Baselines (Lesson 07)

*What you get from distances alone — and the precise point at which it stops being enough.*

| Lesson | What you learn | Key API |
|---|---|---|
| [07a · SchNet](../notebooks/07a_schnet.ipynb) | The continuous-filter convolution and its filter-generating network; the full invariant blueprint from species embedding to atom-wise readout; why energy-conserving forces must come from $\vec F_i = -\partial E/\partial \vec{r}_{i}$ by autograd rather than a separate force head; the standard joint energy + force loss | autograd forces |
| [07b · DimeNet](../notebooks/07b_dimenet.ipynb) | A **runnable counterexample**: two structures no distance-based GNN can tell apart, at any cutoff; directional message passing on *directed edges* using angles, while the prediction stays invariant; the 2D spherical Fourier–Bessel basis and the DimeNet++ efficiency fixes; the $O(Nk^2)$ triplet cost that motivates Part IV | edge/triplet indexing |
| [07c · The rematch: rMD17 aspirin](../notebooks/07c_rmd17_aspirin.ipynb) | The experiment 07b could only promise: the *same* two models, unchanged, on a real molecule whose energy depends on angles, and the comparison **reverses**; working with rMD17 (units, official splits, the 1000-frame rule); an angle-blinding ablation that pins the gap on the angular basis; the $O(Nk^2)$ bill, measured in wall-clock | `load_rmd17` |

## Part IV: State-of-the-Art Equivariant Potentials (Lessons 08-10)

*Three architectures, each as theory then a block-by-block build, all trained on the same
dataset so the comparison is honest.*

| Lesson | What you learn | Key API |
|---|---|---|
| [08a · NequIP theory](../notebooks/08a_nequip_theory.ipynb) | The atomic energy ansatz $E = \sum_i E_i$; the interaction block — convolution filters $R(r_{ij})Y^{(l)}_{m}(\hat r_{ij})$, tensor-product convolution, self-interaction, ResNet update, gate; how chemistry enters; the paper's central claim that $l>0$ features buy **dramatic data efficiency**, and the experiments behind it | — |
| [08b · NequIP implementation](../notebooks/08b_nequip_implementation.ipynb) | Build `SimpleNequIP` end to end; **verify** energy invariance, force covariance and smoothness at the cutoff *before* training; train with the joint energy + force loss; export the checkpoint that Lesson 11 later drives MD with | e3nn blocks, `artifacts/` |
| [09a · Allegro theory](../notebooks/09a_allegro_theory.ipynb) | Why message passing grows the receptive field as $N_\text{layer}\times r_c$ and what that costs at scale; Allegro's strictly **local** pairwise decomposition $E = \sum_{ij} E_{ij}$ with no message passing; the two-track scalar/tensor design and where the tracks couple; the argument behind a 100-million-atom simulation | — |
| [09b · Allegro implementation](../notebooks/09b_allegro_implementation.ipynb) | Build the two-track layer; verify under rotation, **inversion** and translation in float64; train with the same protocol as NequIP; time a forward pass against atom count to *see* strict locality become linear scaling | e3nn blocks |
| [10a · Atomic Cluster Expansion](../notebooks/10a_ace_theory.ipynb) | Body-order expansion and why explicit $K$-body sums cost $\mathcal{O}(N_c^K)$; **the density trick** that collapses this to linear cost via the atomic base $A_{i,nlm}$; symmetrization into the invariant $B$-basis; why the basis is *complete*; the bridge from ACE to message passing | — |
| [10b · MACE theory](../notebooks/10b_mace_theory.ipynb) | ACE's density trick used as the **message function** of an equivariant MPNN; the $A$-basis, higher-order $B$-features, message, update and per-layer readout; body-order accounting — why $\nu=3$ gives 4-body messages and two layers reach effective body order 13; the design-space view placing SchNet, DimeNet, NequIP and MACE in one framework | — |
| [10c · MACE implementation](../notebooks/10c_mace_implementation.ipynb) | Build `SimpleMACE` from those equations; see how `FullyConnectedTensorProduct` realizes the generalized Clebsch–Gordan coupling as *iterated pairwise* contractions, and verify the block is exactly equivariant; train and compare against NequIP and Allegro | e3nn blocks |

## Part V: Applications (Lesson 11)

| Lesson | What you learn | Key API |
|---|---|---|
| [11 · Molecular dynamics with ASE](../notebooks/11_molecular_dynamics_ase.ipynb) | Wrapping a trained potential as an ASE `Calculator`; static energy/force parity on held-out configurations; velocity Verlet and why **NVE energy conservation is *the* sanity check** that forces are consistent with the energy; Langevin NVT sampling compared through $g(r)$; where a toy model breaks and what production workflows add | `ase.md`, `Calculator` |
