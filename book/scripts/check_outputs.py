#!/usr/bin/env python3
"""Report lesson code cells that were never executed.

The book is built with ``execute_notebooks: "off"``, so whatever is committed in each
``.ipynb`` is exactly what readers see. An *unexecuted* cell therefore publishes as a
bare code block with no result -- usually a sign that a notebook was edited but not
re-run.

The signal is ``execution_count``, not the presence of outputs. Plenty of perfectly
healthy cells produce no output at all (``def``, ``class``, plain assignments), so
"has no outputs" massively over-reports; an executed cell always carries an
``execution_count``, whether or not it printed anything.

Also flags cells whose execution counts run *backwards*, which means the notebook was
executed piecemeal and out of order -- the stored outputs may not correspond to a clean
top-to-bottom run.

Usage
-----
    python book/scripts/check_outputs.py              # report, always exit 0
    python book/scripts/check_outputs.py --strict     # exit 1 if anything is unexecuted
    python book/scripts/check_outputs.py --json       # machine-readable summary

Run ``python book/scripts/execute_notebooks.py --missing-only`` to fix what it finds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"


def is_effectively_empty(source: str) -> bool:
    """True for cells that are blank or contain only comments -- nothing to execute."""
    lines = [ln.strip() for ln in source.splitlines()]
    return not [ln for ln in lines if ln and not ln.startswith("#")]


def audit(path: Path) -> dict:
    nb = json.loads(path.read_text())

    total = 0
    unexecuted = []
    silent = 0
    counts = []

    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if is_effectively_empty(source):
            continue

        total += 1
        execution_count = cell.get("execution_count")
        first_line = next((ln for ln in source.splitlines() if ln.strip()), "").strip()

        if execution_count is None:
            unexecuted.append({"cell": index, "preview": first_line[:70]})
        else:
            counts.append(execution_count)
            if cell.get("outputs"): continue

            silent += 1

    out_of_order = counts != sorted(counts)

    return {
        "notebook": path.name,
        "code_cells": total,
        "unexecuted": unexecuted,
        "silent": silent,
        "out_of_order": out_of_order,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero if any cell is unexecuted"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="list every unexecuted cell"
    )
    args = parser.parse_args()

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOK_DIR}", file=sys.stderr)
        return 1

    reports = [audit(p) for p in notebooks]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        total_cells = sum(r["code_cells"] for r in reports)
        total_unexecuted = sum(len(r["unexecuted"]) for r in reports)
        total_silent = sum(r["silent"] for r in reports)

        print(f"Execution audit of {len(reports)} notebooks in notebooks/\n")
        for report in reports:
            n_bad = len(report["unexecuted"])
            flags = []
            if n_bad:
                flags.append(f"{n_bad} unexecuted")
            if report["out_of_order"]:
                flags.append("executed out of order")
            marker = "  " if not flags else "!!"
            note = f"   <- {', '.join(flags)}" if flags else ""
            print(
                f"{marker} {report['notebook']:<45} "
                f"{report['code_cells'] - n_bad:>3}/{report['code_cells']:<3} executed{note}"
            )
            if args.verbose:
                for item in report["unexecuted"]:
                    print(f"       cell {item['cell']:>3}: {item['preview']}")

        print(
            f"\n{total_cells - total_unexecuted}/{total_cells} code cells executed.  ("
            f"{total_silent} of them produce no output, which is normal for def/class/assignment cells.)"
        )

        if total_unexecuted:
            print(
                "\nUnexecuted cells publish as code with no result. To fix:\n"
                "    python book/scripts/execute_notebooks.py --missing-only"
            )

    if args.strict and any(r["unexecuted"] for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
