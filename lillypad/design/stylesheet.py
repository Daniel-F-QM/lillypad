"""Rendering qss/app.qss into an application stylesheet.

`string.Template`, not `str.format` and not an f-string. The stylesheet is made of
CSS braces — ~150 of them — and brace-based templating would need every one
escaped, where a single miss silently unstyles the entire application. Template
ignores braces completely, and QSS has no `$` of its own, so there is nothing to
escape in the other direction either.

`substitute`, never `safe_substitute`: an unknown token should raise at startup,
loudly, rather than ship a rule with a literal `$accent` in it that Qt quietly
discards.
"""

from __future__ import annotations

from string import Template

from lillypad.design.icons_runtime import make_arrow_icons, make_check_icon
from lillypad.design.tokens import METRICS, TYPE
from lillypad.paths import resource_path

# The stylesheet's path, relative to the resource base. It MIRRORS the package
# layout, and Lillypad.spec bundles it to the same relative path, so this one
# tuple is correct from source and frozen alike. A flatter dest in the bundle
# would need the loader to know which of the two it was running in — the kind of
# second opinion about paths that lillypad.paths exists to prevent.
_QSS_PARTS = ("lillypad", "design", "qss", "app.qss")
_template: Template | None = None


def _load_template() -> Template:
    """The stylesheet source, read once.

    Cached because a theme switch re-renders it and the file cannot change under
    a running program.
    """
    global _template
    if _template is None:
        path = resource_path(*_QSS_PARTS)
        _template = Template(path.read_text(encoding="utf-8"))
    return _template


def build_stylesheet(pal, tag) -> str:
    """The application stylesheet for palette `pal`.

    `tag` names the theme and only reaches the generated PNG filenames, so the
    dark and light arrow and tick images do not overwrite each other in %TEMP%.

    The three icons are rendered BEFORE substitution: their paths are tokens in
    the stylesheet, and QSS url() cannot take a data URI.
    """
    arrow_up, arrow_dn = make_arrow_icons(pal["text"], tag)
    check = make_check_icon(pal["bg"], tag)
    return _load_template().substitute(
        **pal, **TYPE, **METRICS,
        arrow_up=arrow_up, arrow_dn=arrow_dn, check=check)


def stylesheet_tokens() -> set[str]:
    """Every placeholder the stylesheet's RULES refer to.

    Comments are stripped and `$$` escapes removed first, so the prose in the
    file's header — which names tokens by example — does not read as a
    requirement. Without that, `$$fs_*` in a comment shows up as a demand for a
    token called `fs_`.
    """
    import re
    src = _load_template().template
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = src.replace("$$", "")
    return set(re.findall(r"\$(\w+)", src))
