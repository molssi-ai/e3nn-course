"""Add theme options that Jupyter Book 1.x does not expose in ``_config.yml``.

WHY THIS EXISTS
---------------
Jupyter Book builds ``html_theme_options`` itself from the ``html:``, ``repository:``
and ``launch_buttons:`` sections of ``_config.yml``. Anything set under
``sphinx.config.html_theme_options`` **replaces** that generated dict wholesale rather
than merging into it -- which silently drops the launch buttons, download buttons,
repository URL and footer. (Found the hard way.)

So instead of restating the whole block and letting it drift out of sync, this extension
merges in only the handful of `sphinx-book-theme` options Jupyter Book has no YAML key
for, at ``config-inited`` time.

It also fills in ``html_context``, which `sphinx-book-theme` 1.3 (via
`pydata-sphinx-theme`) requires for ``use_edit_page_button`` but Jupyter Book 1.0.4 does
not populate -- deriving it from ``repository_url`` so the org/repo/branch are declared
in exactly one place, ``_config.yml``.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Options `sphinx-book-theme` understands but Jupyter Book 1.x has no YAML key for.
EXTRA_THEME_OPTIONS: dict[str, object] = {
    # Sidebar / in-page navigation depth.
    "show_navbar_depth": 1,
    # Show H2s in the right-hand "Contents" panel, not just H1.
    "show_toc_level": 2,
    "home_page_in_toc": True,
    # Needed for the source/edit buttons to resolve to GitHub.
    "repository_provider": "github",
    "use_source_button": True,
    # Distraction-free reading mode.
    "use_fullscreen_button": True,
    # Drop the hamburger `sphinx-book-theme` puts at the start of the article header. Both
    # theme scripts wire up `document.querySelector(".primary-toggle")`, which matches
    # pydata-sphinx-theme's navbar copy -- hidden at every width -- so this button never
    # had a click handler. The sidebar's own Collapse button covers the job.
    "article_header_start": "",
    "icon_links": [
        {
            "name": "MolSSI",
            "url": "https://molssi.org",
            "icon": "fa-solid fa-flask",
            "type": "fontawesome",
        },
        {
            "name": "e3nn documentation",
            "url": "https://docs.e3nn.org",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/molssi-ai/e3nn-course",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}


def _apply(app, config) -> None:
    options = dict(config.html_theme_options or {})
    options.update(EXTRA_THEME_OPTIONS)
    config.html_theme_options = options

    # ``use_edit_page_button`` blows up at render time if ``html_context`` is missing the
    # github_* keys, so derive them from ``repository_url`` rather than restating them.
    if not options.get("use_edit_page_button"):
        return

    repository_url = options.get("repository_url", "")
    parts = [p for p in urlparse(str(repository_url)).path.split("/") if p]
    if len(parts) < 2:
        # Not a parseable org/repo pair -- disable the button instead of failing the build.
        options["use_edit_page_button"] = False
        config.html_theme_options = options
        return

    user, repo = parts[0], parts[1].removesuffix(".git")
    context = dict(config.html_context or {})
    context.setdefault("github_user", user)
    context.setdefault("github_repo", repo)
    context.setdefault("github_version", options.get("repository_branch") or "main")
    # ``doc_path`` is relative to the repo root; Jupyter Book calls it ``path_to_docs``.
    context.setdefault("doc_path", options.get("path_to_docs") or "")
    config.html_context = context


def setup(app):
    app.connect("config-inited", _apply, priority=800)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
