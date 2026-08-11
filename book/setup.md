# Setting up the environment

You do **not** need a local install to read this book: every lesson page shows
its stored outputs, figures and training curves. You need an environment only
when you want to *run* and modify the code on your own machine.

## Local installation

The project is managed with [uv](https://docs.astral.sh/uv/). If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/molssi-ai/e3nn-course.git  # Clone the repository
cd e3nn-course                # Change directory into the repository root

uv sync                       # Create .venv from pyproject.toml + uv.lock
source .venv/bin/activate     # Activate it

uv run jupyter lab            # Launch JupyterLab
```

Every notebook records the standard `python3` kernel, which `uv run jupyter lab`
resolves to this `.venv` automatically. So, there is nothing else to configure.
If you prefer the environment to appear under its own name in the kernel
selector (useful when you juggle several projects), you can register it
explicitly:

```bash
.venv/bin/python -m ipykernel install --user --name e3nn-course \
                 --display-name "Python (e3nn-course)"
```

Requirements: **Python ≥ 3.13** and **PyTorch 2.7.1**. A CUDA GPU is optional:
every training cell in the course is sized to finish in $\leq$ 2 min on a GPU or
$\leq$ 10 min on CPU.

### Verify the installation

Run the following code block in a fresh notebook cell or with `python -c`. It
builds an equivariant tensor product and checks it numerically against a random
rotation:

```python
# From the repository root
import sys; sys.path.insert(0, ".")      
import torch
from e3nn import o3
from course_utils.equivariance import model_equivariance_error

torch.set_default_dtype(torch.float64)

tp = o3.FullyConnectedTensorProduct("1o", "1o", "0e + 1o + 2e")
err = model_equivariance_error(tp, [o3.Irreps("1o"), o3.Irreps("1o")], o3.Irreps("0e + 1o + 2e"))
print(f"torch {torch.__version__} | equivariance error: {err:.2e}")
```

You should see an error around `1e-15`. Anything near `1e-6` means the default
dtype is still `float32`; anything larger means something is genuinely wrong.
Please [open an issue](https://github.com/molssi-ai/e3nn-course/issues) and
provide a minimalistic and reproducible code snippet that generates the error.

## What the environment contains

| Package | Used for |
|---|---|
| `torch` 2.7.1 | tensors, autograd, training |
| `e3nn` | irreps, spherical harmonics, tensor products, gates |
| `torch-geometric` | graph batching and scatter operations |
| `ase` | atomistic structures, neighbor lists, MD integrators (Lesson 11) |
| `matplotlib` | static figures |
| `plotly` | the rotatable 3D figures (spherical harmonics, graphs, trajectories) |
| `sympy` | symbolic Clebsch–Gordan and Wigner algebra |

Shared helpers live in `course_utils/` and are imported by every lesson:

- `course_utils.equivariance`: the numerical equivariance test harness
- `course_utils.plotting`: spherical harmonics, irreps, point clouds, training curves
- `course_utils.data`: small datasets, neighbor lists, train/val splits


## Rebuilding this book locally

The book is built with **Jupyter Book 1.x** from a separate, lightweight
environment. It does *not* need `torch` or `e3nn`, because it renders the
outputs already stored in the notebooks:

```bash
uv venv --python 3.13 .venv-docs # Create a separate environment for the docs
uv pip install --python .venv-docs/bin/python -r book/requirements-docs.txt # Install Jupyter Book
```

Then, from the repository root directory:

```bash
make -C book html      # Build the HTML book at book/_build/html
make -C book serve     # Serve at http://localhost:8000
```

See [How to use this book](how-to-use) for the notebook-execution workflow that
keeps those stored outputs current.
