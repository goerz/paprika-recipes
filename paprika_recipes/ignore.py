"""Markdown files in a recipe directory that are not recipes.

Every `.md` file under a recipe directory is a recipe, which is what lets a
file someone writes from scratch become one on the next push.  But a directory
that lives in a note vault or a git repository usually holds other markdown
too -- a `README.md`, a `CLAUDE.md` of instructions for an assistant, a folder
of drafts -- and without a way to say so, each of those would be uploaded to
Paprika as a recipe of its own.

A `.paprikaignore` file at the top of the directory names them, one pattern
per line, in a deliberately small subset of `.gitignore`'s syntax:

    # Not recipes
    README.md
    CLAUDE.md
    drafts/
    /notes/*.md

-   Blank lines and lines starting with `#` are skipped.
-   A pattern with no `/` in it matches a file or directory of that name at
    any depth.  `*`, `?` and `[...]` work as they do in a shell.
-   A pattern containing a `/` is anchored to the top of the directory, and
    is matched one path component at a time, so `*` never crosses a `/`.  A
    `**` component matches any number of directories.
-   A trailing `/` makes a pattern match only directories, and everything
    inside them.

Negated (`!`) patterns are refused rather than silently ignored: a rule that
quietly did nothing would leave a file being uploaded that its author believed
was excluded, which is the one mistake this file exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Final

from .exceptions import PaprikaUserError

IGNORE_FILENAME: Final = ".paprikaignore"


@dataclass(frozen=True)
class IgnorePattern:
    """One line of a `.paprikaignore` file."""

    #: The pattern split into path components.
    parts: tuple[str, ...]
    #: Whether the pattern is matched from the top of the directory, rather
    #: than against a single name at any depth.
    anchored: bool
    #: Whether the pattern only names directories.
    directory_only: bool

    @classmethod
    def parse(cls, line: str) -> IgnorePattern:
        pattern = line
        directory_only = pattern.endswith("/")
        pattern = pattern.rstrip("/")
        anchored = "/" in pattern
        parts = tuple(part for part in pattern.split("/") if part)

        return cls(parts=parts, anchored=anchored, directory_only=directory_only)

    def matches(self, parts: tuple[str, ...]) -> bool:
        """Does this pattern exclude the file whose relative path is `parts`?

        A pattern that names a directory excludes everything beneath it, so
        each leading run of the path's components is tried in turn; the full
        path is tried last, and only for a pattern that may name a file.
        """
        candidates = range(1, len(parts) if self.directory_only else len(parts) + 1)

        if not self.anchored:
            return any(
                fnmatchcase(parts[index - 1], self.parts[0]) for index in candidates
            )

        return any(_match_components(self.parts, parts[:index]) for index in candidates)


@dataclass(frozen=True)
class IgnoreRules:
    """The parsed contents of a directory's `.paprikaignore`."""

    patterns: tuple[IgnorePattern, ...] = ()

    @classmethod
    def load(cls, root: Path) -> IgnoreRules:
        """Read `root`'s `.paprikaignore`; no file means nothing is ignored."""
        path = root / IGNORE_FILENAME

        if not path.is_file():
            return cls()

        return cls.parse(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def parse(cls, content: str, source: str = IGNORE_FILENAME) -> IgnoreRules:
        patterns = []

        for number, raw in enumerate(content.splitlines(), start=1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("!"):
                raise PaprikaUserError(
                    f"{source}, line {number}: negated patterns ({line!r}) are "
                    "not supported; list only what should be ignored."
                )

            pattern = IgnorePattern.parse(line)

            if not pattern.parts:
                raise PaprikaUserError(
                    f"{source}, line {number}: {line!r} does not name anything."
                )

            patterns.append(pattern)

        return cls(tuple(patterns))

    def __bool__(self) -> bool:
        return bool(self.patterns)

    def ignores(self, relative: Path | PurePosixPath) -> bool:
        """Is the file at `relative` (to the directory's top) excluded?"""
        parts = PurePosixPath(*relative.parts).parts

        return any(pattern.matches(parts) for pattern in self.patterns)


def _match_components(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    """Match path components against pattern components, `**` included."""
    if not pattern:
        return not parts

    head, rest = pattern[0], pattern[1:]

    if head == "**":
        return any(
            _match_components(rest, parts[index:]) for index in range(len(parts) + 1)
        )

    if not parts:
        return False

    return fnmatchcase(parts[0], head) and _match_components(rest, parts[1:])
