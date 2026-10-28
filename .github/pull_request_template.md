<!--
Thank you for contributing! Please open pull requests against `dev`, not `main`.
Keep each pull request to one purpose. See CONTRIBUTING.md for details.
-->

## What and why

<!-- What does this change, and why? Link the issue it resolves, e.g. "Fixes #42". -->

## Type of change

- [ ] Correction or clarification of an existing lesson
- [ ] New or revised exercise
- [ ] New lesson
- [ ] `course_utils` helper
- [ ] Book build, theme or scripts
- [ ] Dependencies or environment
- [ ] Other:

## Lessons affected

<!-- List each notebook this touches, e.g. `10a_ace_theory`, and say whether its
code cells changed (and were re-executed) or only its markdown. -->

## Checklist

- [ ] The branch starts from `dev`, and this pull request targets `dev`.
- [ ] The changed lessons follow [`STYLE_GUIDE.md`](https://github.com/molssi-ai/e3nn-course/blob/main/STYLE_GUIDE.md): equations
      before code, every symbol defined at first use, course-wide notation.
- [ ] Every new citation links a DOI or arXiv abstract page, and every quoted
      equation or section number has been checked against the paper.
- [ ] Every new equivariant operation has a numerical equivariance check.
- [ ] Lessons with changed code were re-executed top to bottom with
      `book/scripts/execute_notebooks.py`, and `book/scripts/check_outputs.py`
      reports nothing for them.
- [ ] New exercises have a difficulty rating, a hidden solution with
      self-contained code, and the real output of that code.
- [ ] The book builds locally (`make -C book html`), and I looked at the
      rendered pages I changed.
- [ ] A new lesson is registered in `book/_toc.yml`, `book/curriculum.md` and
      the README curriculum table, and new papers are in `book/references.bib`.
- [ ] A changed `course_utils` helper: every lesson that uses it was re-executed
      and is listed above.
- [ ] No new dependencies, or `pyproject.toml` and `uv.lock` are updated and the
      reason is explained above.

## Sources and licenses

<!-- Papers (with DOI), reference implementations (with version and license),
and any new figures, data or model files (with source, size and license) that
this change relies on. Write "None" if not applicable. -->
