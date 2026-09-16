# How to use this book

## Reading a lesson

Every lesson follows the same rhythm, so you always know where you are:

1. **Header**: the lesson number and title, a "What you will learn" list, and
   prerequisites;
2. **Setup cell**: imports, seeds, device selection. No hidden magic; you can
   always see exactly what is in scope;
3. **Sections, equations first**: a markdown cell stating the mathematics, then
   a short code cell ($\leq$ 25 lines) implementing the theoretical concept;
4. **Equivariance check**: every new equivariant operation is immediately tested
   numerically;
5. **Visualization**: at least one figure per major concept;
6. **Summary**: what was built, and a pointer to the next lesson;
7. **Exercises**: rated 🌶️ to 🌶️🌶️🌶️ (see [below](#exercise-difficulty)), with
   solutions hidden behind collapsible blocks. Try them yourself first before
   expanding the solution.

### The buttons in the top-right

The four GitHub actions share one {fab}`github` menu; the rest are buttons of their own.

| Button | What it does |
|---|---|
| {fab}`github` **Repository** | Opens the course on GitHub: the source of every lesson, helper and figure. |
| {fas}`code` **Show source** | Opens this page's unrendered source on GitHub: to access the notebook itself rather than the built page. |
| {fas}`pencil` **Suggest edit** | Jumps straight to this page's source on GitHub. |
| {fas}`lightbulb` **Open issue** | Starts a new issue on the course repository, with this page already filled into the title. |
| {fas}`download` **Download** | Grabs the raw `.ipynb`, or renders a printable PDF of the lesson via your browser's print dialog. |
| {fas}`expand` **Fullscreen** | Switch to fullscreen mode. |
| {fas}`circle-half-stroke` **Color mode** | Cycles between your system setting, dark and light; the icon shows the mode in effect. |

:::{tip}
To run a lesson, download it (or clone the repository) and open it from the repository's
`notebooks/` directory, so that the setup cell's `sys.path.insert(0, "..")` puts
`course_utils/` on the path. See [Setting up the environment](setup).
:::

### Interactive figures

Figures rendered with Plotly are **live**: drag to rotate, scroll to zoom,
double-click to reset. These are used where 3D orientation genuinely carries the
meaning: spherical harmonics, atomistic graphs and cutoff spheres, and MD
trajectories. Static Matplotlib figures are used everywhere else, deliberately,
because they are cheaper and print well.

The pages are laid out to use the **full width of your screen**; on a wide
monitor the 3D figures and the tensor-product tables get considerably more room.
Use the fullscreen button for more space.

(exercise-difficulty)=
### Exercise difficulty

Every exercise carries a chilli rating in its title, so you can pick what suits
the time you have:

| | meaning | typically |
|---|---|---|
| 🌶️ | **Easy.** Apply a rule the lesson just stated, or run and read off a result. Often a few lines built on code already in the notebook. | 5-15 minutes |
| 🌶️🌶️ | **Moderate.** Modify the lesson's code and interpret what changes, derive a result and verify it numerically, or run a single training job and explain the outcome. | 30-60 minutes |
| 🌶️🌶️🌶️ | **Hard.** Prove something on paper, build a component that is not in the lesson, or run several experiments and compare them. Expect open-ended reasoning rather than one right number. | an hour or more |

The ratings measure *effort and open-endedness*, not importance. Several of the
three-chilli exercises are the ones that make a lesson's point land hardest, and
the deliberate-sabotage ones (break the envelope, break the symmetry, break the
equivariance) are worth the time precisely because a model that has stopped
being equivariant looks perfectly healthy until you test it.

Timings assume you have read the lesson and are running on the course
environment. Anything involving training is quoted for a GPU, and will take
substantially longer on a CPU.

## Suggested paths through the course

**The full course (recommended).** Lessons 01a $\rightarrow$ 11 in order.
Roughly 15-30 minutes of reading per notebook, plus exercises.

**"I just want to build MACE."** 02a $\rightarrow$ 02b $\rightarrow$ 03a
$\rightarrow$ 03b $\rightarrow$ 04 $\rightarrow$ 06a $\rightarrow$ 10a
$\rightarrow$ 10b $\rightarrow$ 10c. You will miss the invariant-baseline
contrast, but the equivariant machinery is self-contained.

**"I know e3nn, I want the potentials."** Skim Part I, then 05a $\rightarrow$
06a $\rightarrow$ 08a $\rightarrow$ 08b $\rightarrow$ 09a $\rightarrow$ 09b
$\rightarrow$ 10b $\rightarrow$ 10c $\rightarrow$ 11.

**"I'm here for the theory."** The `*a` notebooks mainly focus on the theory,
but other notebooks may offer additional theoretical content as well: 01a, 01b,
02a, 03a, 08a, 09a, 10a, 10b.

## Running and modifying the code

Running the lesson notebooks requires a local Python environment with PyTorch
installed. See [Setting up the environment](setup).

### Keeping stored outputs current

This book renders the outputs **already stored in each `.ipynb`**, rather than
re-executing on every build. That is a deliberate choice: several lessons train
models, and a CPU-only CI runner would need approximately 45 minutes and could
fail the deploy on a single flaky cell. Stored outputs also mean the published
figures are the ones produced on real hardware.

The cost is that outputs can drift from code. To refresh them after editing a lesson:

```bash
# Re-execute one notebook in place (uses the project .venv, so torch + GPU should be available).
python book/scripts/execute_notebooks.py notebooks/04_nonlinearities_and_gates.ipynb

# Re-execute every notebook that has at least one output-less code cell.
python book/scripts/execute_notebooks.py --missing-only

# Re-execute everything (slow: trains all the models).
python book/scripts/execute_notebooks.py --all
```

Then, check nothing was left behind and rebuild:

```bash
# Reports code cells with no stored output
python book/scripts/check_outputs.py

# Rebuilds the HTML book
make -C book html
```

`check_outputs.py` also runs in CI as a non-blocking report, so a pull request
that adds an unexecuted cell is visible in the build log rather than silently
shipping a code block with no result.

## Contributing

The [authoring style guide](style-guide) is the contract every notebook follows:
structure, citation format, the shared math notation, code conventions, and
figure rules. Read it before opening a pull request.

We welcome corrections, clarifications and new exercises. Please open an issue
before starting a pull request, so we can discuss the change and avoid
duplicated work. See:
[github.com/molssi-ai/e3nn-course](https://github.com/molssi-ai/e3nn-course).
