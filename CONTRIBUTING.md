# Contributing to the e3nn course

Thank you for helping improve this course. Corrections, clarifications, new
exercises and lesson proposals are all welcome. Everyone taking part is expected
to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

This guide explains how to report a problem, how to propose new material, and how
to prepare a pull request that can be merged without a long back-and-forth.

## Contents

- [Ways to contribute](#ways-to-contribute)
- [Opening an issue](#opening-an-issue)
- [Setting up a development environment](#setting-up-a-development-environment)
- [Branches and pull requests](#branches-and-pull-requests)
- [Editing a lesson](#editing-a-lesson)
- [Writing exercises](#writing-exercises)
- [Adding a new lesson](#adding-a-new-lesson)
- [Changing `course_utils`](#changing-course_utils)
- [Dependencies](#dependencies)
- [Building the book](#building-the-book)
- [Commit messages](#commit-messages)
- [Licensing and sources](#licensing-and-sources)

## Ways to contribute

| You want to... | Start with |
|---|---|
| Ask about a lesson, the math or the code, or float an idea that is not yet a concrete request | a post in [Discussions](https://github.com/molssi-ai/e3nn-course/discussions) |
| Fix a typo, a wrong equation, a broken link or a misattributed citation | a **Correction or clarification** issue, or a small pull request directly |
| Report a cell that fails, or an output that does not match its text | a **Notebook does not run** issue |
| Propose an exercise for an existing lesson | an **Exercise proposal** issue |
| Suggest a topic the course does not cover | a **Lesson topic request** issue |
| Improve the book's build, theme or scripts | an issue first, then a pull request |

For anything larger than a small fix, **open an issue before writing a pull
request**, so we can agree on the approach and avoid duplicated work. Issues are
for problems and concrete requests; questions are answered in Discussions, and a
discussion that turns up a problem can be converted into an issue.

## Opening an issue

Search the [existing issues](https://github.com/molssi-ai/e3nn-course/issues)
first. Then pick the matching form on the
[new issue page](https://github.com/molssi-ai/e3nn-course/issues/new/choose).
Every lesson page in the [book](https://molssi-ai.github.io/e3nn-course/) also
has an **Open issue** button that fills in the page name for you.

A good issue names the exact place in the course:

- the **notebook** (for example `10a_ace_theory`),
- the **section heading** (for example "Connection to message passing"),
- and, for code problems, the **cell** and the full error message as text, not a
  screenshot.

For a mathematical or citation error, give the source you are comparing against:
the paper, its DOI, and the equation or section number.

## Setting up a development environment

The project is managed with [uv](https://docs.astral.sh/uv/). Two environments
are used, deliberately kept separate:

```bash
git clone https://github.com/molssi-ai/e3nn-course.git
cd e3nn-course

# 1. The project environment: runs the lessons (torch, e3nn, ...)
uv sync

# 2. The docs environment: builds the book only (no torch, no e3nn)
uv venv --python 3.13 .venv-docs
uv pip install --python .venv-docs/bin/python -r book/requirements-docs.txt
```

Open lessons from the `notebooks/` directory, so that the setup cell's
`sys.path.insert(0, "..")` puts `course_utils/` on the path. Full instructions
are in the book's [setup page](https://molssi-ai.github.io/e3nn-course/book/setup.html).

## Branches and pull requests

- **`main`** is the published book. Every push to `main` rebuilds and deploys
  [molssi-ai.github.io/e3nn-course](https://molssi-ai.github.io/e3nn-course/).
- **`dev`** is the integration branch. Maintainers merge `dev` into `main` when a
  set of changes is ready to publish.

To contribute:

1. Branch from `dev` with a short descriptive name, for example
   `fix-10a-summary` or `exercise-06b-chirality`.
2. Keep each pull request to **one purpose**: one lesson, one fix or one
   feature. Small pull requests are reviewed fastest.
3. Open the pull request **against `dev`**, fill in the template, and link the
   issue it resolves (for example `Fixes #42`).
4. Maintainers may push small fixes to your branch before merging.

## Editing a lesson

[`STYLE_GUIDE.md`](STYLE_GUIDE.md) is the contract every notebook follows: the
lesson structure, citation format, math notation, code conventions and figure
rules. Read it before your first pull request. The points that come up most
often in review are:

- **Equations first, then code.** Each section states the math in a markdown
  cell, then implements exactly that math in a short code cell (about 25 lines
  or fewer) whose variable names mirror the symbols.
- **Define every symbol at first use**, and use the course-wide notation from the
  style guide (for example $\vec r_{ij} = \vec r_j - \vec r_i$).
- **Write for the reader.** Each paragraph should follow from the one before it.
  Build on what the lesson has already shown before introducing a new framework
  or paper. Prefer short, complete sentences, and say what a quantity is before
  saying what it costs or why it matters.
- **Cite carefully.** Link every paper to its DOI or arXiv abstract page. Do not
  quote an equation or section number you have not checked against the paper;
  cite the section instead, or leave the number out.
- **Test every new equivariant operation** with
  `course_utils.equivariance.assert_equivariant` (or
  `assert_model_equivariant`) right after it is built.

### Outputs are committed

The book does **not** execute notebooks. It renders the outputs stored in each
`.ipynb`, so those outputs are part of the reviewed content. After changing any
code cell:

```bash
# Re-execute the lesson top to bottom in a fresh kernel (project environment)
.venv/bin/python book/scripts/execute_notebooks.py 10a

# Confirm no cell was left unexecuted or run out of order
.venv-docs/bin/python book/scripts/check_outputs.py
```

Do not commit a notebook that was run piecemeal: the stored outputs must come
from one clean top-to-bottom run. A markdown-only change does not need a re-run.

Training cells have a time budget (see the style guide). Quote numbers only to
the precision a re-run reproduces: GPU scatter-adds are nondeterministic, so
seeded runs can drift by a few tenths of a percent.

## Writing exercises

Exercises go in the lesson's **Exercises** cell, after the Summary. Each one has:

1. **A title with a difficulty rating**, in the form
   `**N. Title (Difficulty: 🌶️):**`. The scale (🌶️ easy, 🌶️🌶️ moderate,
   🌶️🌶️🌶️ hard) is described in the book's
   [How to use this book](https://molssi-ai.github.io/e3nn-course/book/how-to-use.html#exercise-difficulty)
   page.
2. **A statement that poses a task without giving away the answer.** State the
   premise and ask the reader to explain, derive or verify the result.
3. **A hidden solution** in a `<details><summary>Solution</summary>` block,
   containing, in order:
   - the reasoning, written out step by step;
   - a `python` code block that is **self-contained**: it may use the lesson's
     helper functions and inputs (such as `atomic_base` or `vecs`), but must
     build every other quantity it needs itself;
   - an **Output** block (` ```text `) with the output of that code, pasted
     from an actual run.

Before opening the pull request, run each solution against a kernel that has
executed the lesson, and check that the pasted output matches. If you change the
lesson's inputs or seed later, update the pasted outputs too.

## Adding a new lesson

Open a **Lesson topic request** issue first and agree on the scope with the
maintainers. Then:

1. **Name the notebook** `NN[a|b|c]_short_title.ipynb`. Theory and
   implementation are split into `a`/`b` (and `c`) notebooks for the larger
   models, and every sub-notebook has its own setup cell.
2. **Follow the lesson structure** in the style guide: header, setup cell,
   sections, equivariance checks, figures, summary, exercises. Aim for at most
   about 35 cells and 15 minutes of reading.
3. **Register it in four places:**
   - [`book/_toc.yml`](book/_toc.yml): one `- file:` line under the right part;
   - [`book/curriculum.md`](book/curriculum.md): the lesson row;
   - [`README.md`](README.md): the "Curriculum at a Glance" table;
   - [`book/references.bib`](book/references.bib) and the README's
     "Primary References", for any new paper.
4. **Artifacts.** If a later lesson needs a trained model or a cached dataset,
   export it to `notebooks/artifacts/` and keep it small. Mention the file and
   its size in the pull request.

## Changing `course_utils`

`course_utils/` holds the helpers every notebook imports. Never copy helper code
into a notebook; add or fix it here instead.

A change to a helper can change the output of every lesson that uses it. Find
those lessons with `grep -l "<helper name>" notebooks/*.ipynb`, re-execute them,
and list them in the pull request.

## Dependencies

The environment is fully specified by `pyproject.toml` and `uv.lock`. Notebooks
must never contain `!pip install` cells.

PyTorch is pinned to `2.7.1` and the optional cuEquivariance extra to `0.8.1`,
because the committed outputs were produced with them. Do not upgrade either
without discussing it in an issue first. If you add a dependency, run `uv lock`,
commit the updated `uv.lock`, and explain in the pull request why it is needed.

## Building the book

```bash
make -C book html      # build to book/_build/html
make -C book serve     # build, then serve at http://localhost:8000
make -C book check     # report unexecuted lesson cells
```

The `book` workflow builds the book on every pull request to `main`. Build it
locally before opening a pull request that touches a lesson, `book/` or
`STYLE_GUIDE.md`, and look at the rendered pages you changed.

## Commit messages

Write short, lowercase summaries in the present tense, and refer to lessons by
their number:

```text
adds L09b to TOC
fixes the references in L09a
starts polishing the exercises of L10a
```

## Licensing and sources

The course is released under the [MIT License](LICENSE.md), and contributions
are accepted under the same license.

- Write explanations in your own words. Short quotations from papers are fine
  when they are marked as quotations and cited.
- Implement models from their papers and documented behavior. Do not copy code
  from other projects.
- Only include figures, data or model files whose license allows
  redistribution, and state their source and license in the pull request.

If you use the course in your research, please cite it as described in the
[README](README.md#citing-this-course) or [`CITATION.cff`](CITATION.cff).
