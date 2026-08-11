# Equivariant Graph Neural Networks with e3nn: A Hands-On Course

This course offers a comprehensive, step-by-step and user-friendly tutorial series
which takes you from the mathematical foundations of Euclidean symmetry to complete
and working implementations of state-of-the-art invariant baseline models **SchNet**,
**DimeNet** and equivariant interatomic potentials such as **NequIP**, **Allegro**,
and **MACE**.

:::{admonition} Under active development :class: note Lessons are being written
and revised continuously. If something is unclear, wrong, or missing, please
[open an issue](https://github.com/molssi-ai/e3nn-course/issues). Your feedback
is important to us. 

:::

## The teaching philosophy

Five commitments shape every lesson in this book.

**Theory first.** Each operation is derived and motivated mathematically
*before* presenting any code. We chose to trade some mathematical rigor for user
friendliness but we still maintain a strong connection to theory and attempt to
cite the key manuscripts.

**Theory and experiment, side-by-side.** Implementations are broken into code
blocks of small to moderate sizes while being connected to their corresponding
equations.

**Verifications.** Every key equivariant operation is unit-tested numerically,
using shared helper functions from the local
[`course_utils`](https://github.com/molssi-ai/e3nn-course/blob/main/course_utils/equivariance.py)
folder. Feel free to inspect and modify them at your own convenience! If a block
claims to be equivariant, the notebook proves it to $\approx10^{-15}$.

**Visualizations.** Spherical harmonics, tensor-product selection rules, learned
features, training curves, and MD trajectories are plotted throughout to assist
the reader in building intuition and gaining visual insight: several of them are
designed as interactive 3D figures.

**Modularity and simplicity.** Long lessons are split into `*_a`, `*_b`, `*_c`
notebooks that can be followed in one sitting, individually.

## What you will build

By the end of the course, you will develop a working understanding of
$E(3)$-equivariant models and the e3nn library, and will be able to implement:

- an equivariant point convolution (Tensor Field Networks) that passes a numerical
  $\Og{3}$-equivariance test;
- a small equivariant GNN that classifies **chiral** 3D shapes: something no
  distance-only model can do;
- **SchNet** and **DimeNet++** as invariant baselines, including a concrete
  demonstration of the incompleteness of distance-only descriptors;
- **NequIP**, **Allegro** and **MACE**, block by block, trained on small datasets;
- an ASE calculator wrapping a trained potential, driving real molecular
  dynamics with energy-conservation and radial distribution function (RDF)
  sanity checks.

## Prerequisites

- **Linear algebra** (matrices, eigenvalues, change of basis), basic group theory helps but
  is introduced from scratch in Lesson 01.
- **PyTorch** basics (tensors, autograd, `nn.Module`, training loops).
- Some exposure to **molecular systems / atomistic simulation** is helpful for Parts
  III–V but not required.

## Start here

`````{grid} 1 1 2 2
:gutter: 3

````{grid-item-card} 🚀 Set up the environment
:link: setup
:link-type: doc

Install the course environment with `uv`, register the Jupyter kernel, and check your
install with a one-line equivariance test.
````

````{grid-item-card} 🧭 How to use this book
:link: how-to-use
:link-type: doc

How each lesson is structured, suggested paths through the course, and how to keep the
stored notebook outputs current.
````

````{grid-item-card} 🗺️ Curriculum at a glance
:link: curriculum
:link-type: doc

All twenty-one lessons in five parts, with what each one teaches and the e3nn API it
introduces.
````

````{grid-item-card} 📖 Lesson 01a: Symmetry and equivariance
:link: ../notebooks/01a_symmetry_and_equivariance
:link-type: doc

Jump straight into the first lesson: why symmetry is the right inductive bias, and why
data augmentation is not enough.
````
`````

## Citing this course

If this material is useful in your teaching or research, please cite the repository and
the primary papers for the architectures you use: all collected on the
[Further reading](further-reading) page with DOIs.
