#!/usr/bin/env python3
"""Execute lesson notebooks in place, so the committed outputs are the published ones.

The book is built with ``execute_notebooks: "off"``: whatever is stored in each ``.ipynb``
is exactly what readers see. That keeps the docs build fast and reproducible, but it means
the outputs are a *committed artifact* -- editing a cell without re-running it publishes a
code block with no result. ``check_outputs.py`` finds those; this script fixes them.

Notebooks are executed top-to-bottom in a fresh kernel and written back to the same path,
so a clean run is what lands in git. Run it from the *project* environment (`uv sync`),
not the docs environment -- executing the lessons needs torch and e3nn.

Usage
-----
    python book/scripts/execute_notebooks.py --all              # everything (slow)
    python book/scripts/execute_notebooks.py --missing-only     # only what needs it
    python book/scripts/execute_notebooks.py notebooks/04_nonlinearities_and_gates.ipynb
    python book/scripts/execute_notebooks.py 04 07b             # or match by substring
    python book/scripts/execute_notebooks.py --dry-run          # list, do not execute

Run ``python book/scripts/check_outputs.py`` afterwards to confirm nothing is left.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from check_outputs import NOTEBOOK_DIR, ROOT, audit


def select(targets: list[str], missing_only: bool) -> list[Path]:
    """Lesson notebooks to execute, in lesson order.

    A target is either a path to a notebook or a substring of its filename, so both
    `notebooks/04_nonlinearities_and_gates.ipynb` and a bare `04` work.
    """
    paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if targets:
        named = {Path(t).resolve() for t in targets if Path(t).is_file()}
        words = [t for t in targets if not Path(t).is_file()]
        paths = [
            p for p in paths
            if p.resolve() in named or any(w in p.name for w in words)
        ]
    if missing_only:
        # An out-of-order notebook is stale too: its stored outputs may not correspond to
        # any single clean top-to-bottom run.
        paths = [
            p
            for p in paths
            if (report := audit(p))["unexecuted"] or report["out_of_order"]
        ]
    return paths


def execute(path: Path, timeout: int, kernel: str | None) -> None:
    """Run one notebook in a fresh kernel and write it back in place."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel or nb.metadata.get("kernelspec", {}).get("name", "python3"),
        # Relative imports (`sys.path.insert(0, "..")`) and artifact paths in the lessons
        # assume the notebook's own directory is the working directory.
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(nb, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="notebook paths, or substrings of their filenames (default: all lessons)",
    )
    parser.add_argument(
        "--all", action="store_true", help="re-execute every lesson (the default)"
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="skip notebooks that are already cleanly executed",
    )
    parser.add_argument(
        "--timeout", type=int, default=1200, help="per-cell timeout in seconds"
    )
    parser.add_argument("--kernel", help="override the kernel recorded in the notebook")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would run, then stop"
    )
    args = parser.parse_args()

    if args.all and (args.targets or args.missing_only):
        parser.error("--all cannot be combined with targets or --missing-only")

    paths = select(args.targets, args.missing_only)
    if not paths:
        target = "notebooks need executing" if args.missing_only else "notebooks matched"
        print(f"Nothing to do: no {target}.")
        return 0

    print(f"Executing {len(paths)} notebook(s) from {NOTEBOOK_DIR.relative_to(ROOT)}/\n")
    if args.dry_run:
        for p in paths:
            print(f"   {p.name}")
        return 0

    failed = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name} ... ", end="", flush=True)
        start = time.perf_counter()
        try:
            execute(path, args.timeout, args.kernel)
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"FAILED after {time.perf_counter() - start:.0f}s")
            print(f"        {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(path.name)
        else:
            print(f"ok ({time.perf_counter() - start:.0f}s)")

    if failed:
        print(f"\n{len(failed)} notebook(s) failed to execute:", file=sys.stderr)
        for name in failed:
            print(f"   {name}", file=sys.stderr)
        return 1

    print("\nAll notebooks executed. Verify with:\n    python book/scripts/check_outputs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
