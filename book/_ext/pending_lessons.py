"""Let the curriculum keep every lesson listed, whether or not it is published yet.

The curriculum page is the full map of the course: all twenty-one lessons, in five parts,
each linking to its notebook. Lessons are released in batches, so for part of the course's
life some of those notebooks are not in the repository yet.

Rather than editing the page every time a lesson lands, the table always links to every
lesson and this extension decides at build time which links can work. A link whose notebook
is in the build becomes an ordinary link; one whose notebook is absent is rendered as plain
text with a "coming soon" marker. Adding the notebook -- to the repository and to
``_toc.yml`` -- is all it takes for its link to go live; the page itself never changes.

Without this, MyST would still drop the unresolvable links to plain text, but would warn
``myst.xref_missing`` once per unpublished lesson, burying real broken links in noise. The
transform therefore runs before MyST's resolver (priority 9) and consumes those references
first.

Only links under ``notebooks/`` are treated this way, so a genuine typo in any other
cross-reference still fails loudly.
"""

from __future__ import annotations

import posixpath
import re

from docutils import nodes
from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform

#: Lessons live here, and only lesson links may be "not published yet".
LESSON_DIR = "notebooks"

#: A lesson file name: two digits, an optional letter, then the topic (``03b_....ipynb``).
LESSON_FILE = re.compile(r"^\d{2}[a-z]?_[a-z0-9_]+\.ipynb$")

#: Appended after the lesson name, and styled by book/_static/custom.css.
MARKER_TEXT = "coming soon"


def lesson_docname(target: str) -> str | None:
    """The document a lesson link points at, or None if it is not a link to a lesson.

    Targets reach this transform in two shapes. A published lesson resolves at parse time
    to a docname (``notebooks/03b_tensor_products_e3nn``). An unpublished one cannot be
    resolved, so it survives as the path exactly as written -- ``../notebooks/04_x.ipynb``
    from a book page, or a bare ``06a_y.ipynb`` from a sibling notebook. Both are reduced
    to the docname the lesson would have once it is published.
    """
    path = target.split("#", 1)[0].strip()
    if not path:
        return None
    if not path.endswith(".ipynb"):
        # Already a docname; only treat it as a lesson if it lives in the lessons folder.
        return path if path.startswith(f"{LESSON_DIR}/") else None
    name = posixpath.basename(path)
    return f"{LESSON_DIR}/{name[: -len('.ipynb')]}" if LESSON_FILE.match(name) else None


class MarkUnpublishedLessons(SphinxPostTransform):
    # Must beat myst_parser's MystReferenceResolver (priority 9), which would otherwise
    # resolve these references first and warn about every one of them.
    default_priority = 5

    def run(self, **kwargs) -> None:
        for node in list(self.document.findall(addnodes.pending_xref)):
            if node.get("reftype") != "myst":
                continue

            docname = lesson_docname(node.get("reftarget") or "")
            if docname is None or docname in self.env.all_docs:
                # Not a lesson link, or the lesson is published: let MyST link it.
                continue

            node.replace_self(self._as_plain_text(node))

    @staticmethod
    def _as_plain_text(node: addnodes.pending_xref) -> nodes.Node:
        """Render the link's caption as unlinked text, plus the "coming soon" marker."""
        wrapper = nodes.inline("", "", classes=["lesson-pending"])
        # node[0] holds the link text; keep it exactly as written in the table.
        wrapper.extend(node[0].deepcopy().children)
        wrapper += nodes.inline(
            "", MARKER_TEXT, classes=["lesson-pending-marker"]
        )
        return wrapper


def setup(app):
    app.add_post_transform(MarkUnpublishedLessons)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
