"""Number every display equation written as ``$$ ... $$``.

Sphinx's ``math_number_all`` is implemented inside the reStructuredText ``math``
directive, so it never reaches equations written in MyST/Markdown: those arrive as
``math_block`` nodes with no label, and unlabelled equations are not numbered. The lessons
are written in Markdown, so without this the setting does nothing.

This applies the same steps the directive would: give each unlabelled block a generated
label, register it with the math domain, and record the number the domain hands back.
Numbering restarts on every page, because the serial counter lives in per-document
temporary data.

Equations that already carry a label are left alone, so an explicit ``$$ ... $$ (eq:foo)``
keeps working and can still be referenced with ``{eq}`foo```.
"""

from __future__ import annotations

from docutils import nodes
from sphinx.transforms import SphinxTransform


class NumberDisplayEquations(SphinxTransform):
    # After the Markdown has been turned into a doctree, while still in the read phase, so
    # the numbers are stored in the environment like any other equation's.
    default_priority = 800

    def apply(self, **kwargs) -> None:
        if not self.config.math_number_all:
            return

        domain = self.env.get_domain("math")
        for node in self.document.findall(nodes.math_block):
            if node.get("nowrap") or node.get("label"):
                continue
            serial = self.env.new_serialno("sphinx.ext.math#equations")
            node["label"] = f"{self.env.docname}:{serial}"
            node["docname"] = self.env.docname
            domain.note_equation(self.env.docname, node["label"], location=node)
            node["number"] = domain.get_equation_number_for(node["label"])


def setup(app):
    app.add_transform(NumberDisplayEquations)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
