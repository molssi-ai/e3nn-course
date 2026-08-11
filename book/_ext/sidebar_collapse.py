"""Keep the Collapse Sidebar button on every page.

pydata-sphinx-theme's ``layout.html`` drops ``sidebar-collapse.html`` together with its
own ``sidebar-nav-bs.html`` whenever ``suppress_sidebar_toctree()`` is true -- pages with
no ancestor in the toctree, which here means the root page, ``genindex`` and ``search``.
The reasoning is that without a navigation tree there is nothing to collapse.

That does not hold for this book: the navigation comes from sphinx-book-theme's
``sbt-sidebar-nav.html``, which is not in the reject list and renders on those pages
regardless. So the nav appeared with no way to collapse it. Forcing the check to false
keeps the button, and touches nothing else -- the flag is read in exactly one place, and
only to filter those two template names.
"""

from __future__ import annotations

NAV_TEMPLATE = "sidebar-collapse.html"


def _keep_collapse_button(app, pagename, templatename, context, doctree) -> None:
    if NAV_TEMPLATE not in (context.get("sidebars") or []):
        return
    if not callable(context.get("suppress_sidebar_toctree")):
        return

    def suppress_sidebar_toctree(*args, **kwargs) -> bool:
        return False

    context["suppress_sidebar_toctree"] = suppress_sidebar_toctree


def setup(app):
    # After pydata's own html-page-context handlers, which install the original callable.
    app.connect("html-page-context", _keep_collapse_button, priority=900)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
