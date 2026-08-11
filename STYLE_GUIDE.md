# Course Style Guide

This is the contract every notebook in the course follows. It is reproduced from
[`STYLE_GUIDE.md`](https://github.com/molssi-ai/e3nn-course/blob/main/STYLE_GUIDE.md)
at the repository's root directory.

## Structure of a lesson

1. **Header cell** (markdown): lesson number + title, a 3–6 bullet "What you will learn",
   a "Prerequisites" line pointing to earlier lessons, and the primary references for this
   lesson (with equation numbers where applicable).
2. **Setup cell** (code): imports, `torch.manual_seed(0)`, device selection
   (`cuda` if available), and `course_utils` imports. Keep it minimal; no hidden magic.
3. **Sections**: each section is *equations first, then code*:
   - A markdown cell deriving/stating the math with LaTeX. Define every symbol at first use.
   - A short code cell (≲ 25 lines) implementing exactly that math. Variable names mirror
     the math (`l_max`, `Y_lm`, `w_path`).
   - A markdown or inline-comment mapping: "line X ↔ Eq. (Y)".
4. **Equivariance check**: any new equivariant operation gets a numerical test via
   `course_utils.equivariance.assert_equivariant` immediately after it is built.
5. **Visualization**: at least one figure per major concept (matplotlib for statics,
   plotly only where 3D interactivity genuinely helps).
6. **Summary cell**: bullets of what was built + pointer to the next lesson.
7. **Exercises** (optional but encouraged): 2–3 short exercises with hidden solutions
   (`<details>` blocks).

## Citations

- Cite papers as: *Author et al., Title (Venue Year), Eq. (N)* — and link the
  canonical record: a DOI, or the arXiv abstract page.
- Do not invent equation numbers. If unsure, cite the section instead.

## Math notation (used consistently across the whole course)

- Group elements: $g \in O(3)$, rotations $R \in SO(3)$; parity/inversion $P$.
- Wigner D-matrix: $D^{(l)}(g)$; irrep label $(l, p)$ with parity $p \in \{e,
  o\}$.
- Spherical harmonics: $Y^{(l)}_m(\hat{r})$; unit vectors carry a hat.
- Node features: $h_i$ (layer index as superscript $h_i^{(t)}$); edges $(i, j)$
  with $\vec{r}_{ij} = \vec{r}_j - \vec{r}_i$ (state the sign convention
  whenever used!).
- Tensor product with weights: $x \otimes_w y$; Clebsch–Gordan coefficients
  $C^{(l_3,m_3)}_{(l_1,m_1)(l_2,m_2)}$.

## Code conventions

- Python $\geq$ 3.13, PyTorch $\geq$ 2.7, e3nn 0.6.x APIs (`e3nn.o3`,
  `e3nn.nn`).
- Notebooks import shared helpers from `course_utils` (`sys.path.append("..")`
  in setup cell). Never copy-paste helper code between notebooks.
- Every trainable model prints its parameter count after construction.
- Training cells must run in $\leq$ $\approx$2 minutes on GPU / $\leq$
  $\approx$10 on CPU; datasets are kept tiny (aspirin/MD17-style subsets, toy
  point clouds). **Reproducibility over benchmark accuracy.**
- Set seeds; keep any download optional with a local fallback.
- No `!pip install` cells! The environment is fully specified by
  `pyproject.toml`.

## Modularity

- Target $\leq$ $\approx$35 cells / $\leq$ $\approx$15 min reading per notebook.
  If a lesson outgrows that, split into `NNa_`, `NNb_`, `NNc_` sub-notebooks,
  each self-contained (with their own setup cells).
- Theory (`*a_...`) and implementation (`*b_...`) split is preferred for the big models.

## Figures

- Label axes with units; add titles; `fig.tight_layout()`.
- For spherical harmonics / equivariant features, prefer surface plots colored by sign
  (red/blue) as on e3nn.org.
- Keep figures reproducible (seeded) and cheap to render.
