"""Spelling a recipe's categories by name rather than by uid.

Paprika's sync API stores a recipe's categories as a list of category uids;
the names live in a separate list the account keeps.  A file that showed the
uids would be unreadable to its owner and unwritable by hand, so a recipe
directory keeps a copy of that list in `.paprika/categories.json` and the
document format translates in both directions: uids are written out as names,
and names typed into a file are read back as uids.

Everything outside the file -- base copies, merging, uploads -- keeps working
in uids, which is what makes renaming a category in the app a cheap event
rather than an edit to every recipe in it: see `Repository.update_categories`.

Two cases fall back to the uid, so that the translation never loses anything:

-   A uid the list does not know (a category deleted in the app while recipes
    still name it) has no name to show, so the uid itself is written.
-   A name that two categories share -- possible with nested categories --
    could not be read back unambiguously, so both are written by uid.

And one case passes through untouched: a name that matches no category is kept
as typed.  Reading never fails over it, since `status` and `pull` must still
work, but `push` refuses to upload a recipe naming a category Paprika does not
have; see `CategoryNames.unknown`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategoryNames:
    """An account's categories, by uid."""

    names: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_api(cls, items: Iterable[Mapping[str, Any]]) -> CategoryNames:
        """Read the `result` list of Paprika's `/sync/categories/` endpoint."""
        return cls(
            {
                str(item["uid"]): str(item.get("name") or "")
                for item in items
                if item.get("uid") and not item.get("deleted")
            }
        )

    @classmethod
    def load(cls, path: Path) -> CategoryNames:
        """Read a saved copy; a missing or unreadable one knows no categories."""
        if not path.is_file():
            return cls()

        try:
            with open(path, encoding="utf-8") as inf:
                data = json.load(inf)
        except ValueError:
            return cls()

        if not isinstance(data, dict):
            return cls()

        return cls({str(uid): str(name) for uid, name in data.items()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as outf:
            json.dump(dict(self.names), outf, indent=2, sort_keys=True)
            outf.write("\n")

    def __bool__(self) -> bool:
        return bool(self.names)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoryNames):
            return NotImplemented

        return dict(self.names) == dict(other.names)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.names.items())))

    @property
    def _by_name(self) -> dict[str, str]:
        """Names that identify exactly one category, mapped to its uid."""
        counts = Counter(self.names.values())

        return {
            name: uid
            for uid, name in self.names.items()
            if name and counts[name] == 1 and name not in self.names
        }

    def display(self, value: str) -> str:
        """How to write the category `value` (a uid) in a file."""
        name = self.names.get(value)

        if name is not None and self._by_name.get(name) == value:
            return name

        return value

    def resolve(self, value: str) -> str:
        """Which category a value read from a file means: its uid if known."""
        if value in self.names:
            return value

        return self._by_name.get(value, value)

    def display_all(self, values: Iterable[Any]) -> list[Any]:
        return [self.display(v) if isinstance(v, str) else v for v in values]

    def resolve_all(self, values: Iterable[Any]) -> list[Any]:
        return [self.resolve(v) if isinstance(v, str) else v for v in values]

    def unknown(self, values: Iterable[Any]) -> list[str]:
        """The values that name no category this account has."""
        return [str(value) for value in values if str(value) not in self.names]
