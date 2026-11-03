"""A working directory of recipe markdown files, plus the state needed to sync it.

The layout mirrors the one idea we are borrowing from git: alongside the files
you edit, we keep a pristine copy of every recipe exactly as it arrived from
the server.

    <root>/
        .paprika/
            config.yaml             -- which account this directory belongs to
            categories.json         -- the account's category names, by uid
            recipes/<uid>.json      -- the recipe as last pulled ("the base")
            photos/<uid>.json       -- which photo we last downloaded, and
                                       a digest of the bytes we wrote
        attachments/<photo>         -- recipe photos, linked from the files
        <Recipe Name>.md            -- the file you actually edit

The base copies earn their keep twice over.  They are the merge base when both
sides of a sync have moved, and -- more importantly -- they are what makes
change detection reliable.

We decide whether a file has been edited by rendering its base copy and
comparing the resulting *text* against the file on disk, rather than by parsing
the file and comparing fields.  The distinction matters because it makes
correctness independent of how faithful our markdown round-trip is: both sides
of the comparison pass through the identical transformation, so any
normalisation the renderer performs cancels out.  Parsing only ever happens for
files we have already established were edited, which means an imperfect
round-trip can cost us accuracy in a field the user deliberately changed, but
can never silently rewrite the rest of the recipe or manufacture a spurious
upload.

Note also that we never recompute Paprika's `hash` to detect local edits.  For
any recipe with a photo the app folds the image into that value in a way we
cannot reproduce, so it is only meaningful as an opaque token for asking
whether the *server's* copy has moved.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Container, Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar

from .categories import CategoryNames
from .constants import DEFAULT_DOMAIN
from .exceptions import PaprikaProgrammingError, PaprikaUserError
from .ignore import IGNORE_FILENAME, IgnoreRules
from .markdown import (
    ATTACHMENTS_DIRNAME,
    GALLERY_FIELD,
    PHOTO_FIELD,
    DocumentFormat,
    Extras,
    documents_differ,
    normalize_recipe,
    parse_recipe,
)
from .merge import IGNORED_FIELDS, has_conflict_markers
from .recipe import BaseRecipe
from .remote import RemoteRecipe
from .utils import dump_yaml, load_yaml

T = TypeVar("T", bound=BaseRecipe)

REPOSITORY_DIRNAME: Final = ".paprika"
BASE_DIRNAME: Final = "recipes"
PHOTOS_DIRNAME: Final = "photos"
CONFIG_FILENAME: Final = "config.yaml"
CATEGORIES_FILENAME: Final = "categories.json"
RECIPE_SUFFIX: Final = ".md"

#: Characters we refuse to put in a filename.  Paprika itself is happy with
#: any of them, but they are either illegal or a nuisance somewhere we might
#: reasonably expect a recipe directory to be synced to.
UNSAFE_FILENAME_CHARACTERS: Final = '/\\:*?"<>|'


class Status(Enum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"


@dataclass(frozen=True)
class WorkingRecipe:
    """A recipe file in the working directory, and how it relates to its base."""

    uid: str
    status: Status
    path: Path | None
    recipe: RemoteRecipe | None
    base: RemoteRecipe | None
    #: What the file holds that Paprika has nowhere to put.
    extras: Extras = field(default_factory=Extras)
    #: The embed still names the same photo, but the bytes in `attachments/`
    #: are no longer the ones we downloaded: the user swapped the image out
    #: in place.  Invisible to a text comparison of the file, so it is
    #: detected separately; see `Repository._photo_replaced`.
    photo_replaced: bool = False
    #: Gallery files whose bytes the user swapped out in place, by filename;
    #: the gallery's counterpart of `photo_replaced`.
    gallery_replaced: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        """What to call this recipe when talking to the user about it."""
        for recipe in (self.recipe, self.base):
            if recipe is not None and recipe.name:
                return recipe.name

        return self.uid

    def changed_fields(self) -> list[str]:
        """Which fields the user actually edited, relative to the base copy."""
        if self.recipe is None or self.base is None:
            return []

        current = self.recipe.as_dict()
        base = self.base.as_dict()

        changed = {
            key
            for key in current
            # `hash` is the server's token, not an edit, and the photo's
            # bookkeeping never enters the file to be edited.  See
            # `merge.IGNORED_FIELDS`, which is the same judgement.
            if key not in IGNORED_FIELDS and current[key] != base.get(key)
        }

        if self.photo_replaced:
            # The embed reads the same, but the image behind it does not.
            changed.add(PHOTO_FIELD)

        if self.gallery_replaced:
            changed.add(GALLERY_FIELD)

        return sorted(changed)

    @property
    def untracked(self) -> bool:
        """Is this a recipe file that has never been given an identity?

        A recipe someone wrote themselves, in other words: it is on its way
        to Paprika but has not been there yet, so there is no uid in its
        frontmatter to say which recipe it is.  It gets one at the moment it
        is pushed; see `Repository.adopt`.
        """
        return not self.uid

    @property
    def conflicted(self) -> bool:
        """Is a merge still half-resolved in this file?"""
        return self.recipe is not None and has_conflict_markers(self.recipe)

    def has_local_changes(self) -> bool:
        """Is there anything here the server would care about?

        A file can be MODIFIED without this being true: the user may have
        reordered the frontmatter or reflowed a list, which changes the bytes
        on disk without changing the recipe.  Sync decisions key off this
        rather than off `status` alone, so that cosmetic edits neither
        manufacture an upload nor stand in the way of one.
        """
        if self.status in (Status.ADDED, Status.DELETED):
            return True

        return bool(self.changed_fields())


@dataclass(frozen=True)
class GalleryPhotoState:
    """What we know about one gallery photo we wrote into `attachments/`.

    Everything but `content_hash` is the server's record of the photo, kept
    so that a push can address it (`uid`) and tell what changed (`name`,
    `order_flag`, `hash`).  `content_hash` digests the bytes we wrote, which
    is how a file the user replaced in place is noticed.
    """

    uid: str = ""
    name: str = ""
    order_flag: int = 0
    hash: str = ""
    content_hash: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GalleryPhotoState:
        return cls(
            uid=str(data.get("uid", "")),
            name=str(data.get("name", "") or ""),
            order_flag=int(data.get("order_flag", 0) or 0),
            hash=str(data.get("hash", "") or ""),
            content_hash=str(data.get("content_hash", "")),
        )


@dataclass(frozen=True)
class PhotoState:
    """What we know about a recipe's photos as we last wrote them.

    `photo` and `photo_hash` are the server's: the filename of the recipe's
    thumbnail and the token that says which version of it that is.  Together
    they are what lets a pull decide whether the attachment on disk is
    already the thumbnail the server holds.  `content_hash` is ours: a digest
    of the thumbnail bytes we wrote.

    `gallery` records every gallery photo we mirrored, by filename, and
    `large` names the one among them that is the recipe's own full-size
    photo -- the file the embed beneath the title shows.
    """

    photo: str = ""
    photo_hash: str = ""
    content_hash: str = ""
    large: str = ""
    gallery: dict[str, GalleryPhotoState] = field(default_factory=dict)

    def displayed(self) -> tuple[str, str]:
        """The full-size file the photo embed links to (else the thumbnail),
        and the digest of its bytes."""
        if self.large and self.large in self.gallery:
            return self.large, self.gallery[self.large].content_hash

        return self.photo, self.content_hash


@dataclass
class RepositoryConfig:
    account: str = ""
    domain: str = DEFAULT_DOMAIN
    #: Prepended to every frontmatter field we own; see `DocumentFormat`.
    #: Chosen when the directory is cloned, because changing it afterwards
    #: would mean rewriting every file in it.
    frontmatter_prefix: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryConfig:
        return cls(
            account=data.get("account", ""),
            domain=data.get("domain") or DEFAULT_DOMAIN,
            frontmatter_prefix=data.get("frontmatter_prefix") or "",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "domain": self.domain,
            "frontmatter_prefix": self.frontmatter_prefix,
        }


class Repository:
    _root: Path
    _config: RepositoryConfig | None
    _format: DocumentFormat | None

    def __init__(self, root: Path):
        self._root = root
        self._config = None
        self._format = None

        if not self.repository_dir.is_dir():
            raise PaprikaUserError(
                f"{root} is not a paprika recipe directory; "
                "run `paprika-recipes clone` to create one."
            )

    # -- Locating and creating repositories ---------------------------------

    @classmethod
    def initialize(
        cls, root: Path, config: RepositoryConfig | None = None
    ) -> Repository:
        base_dir = root / REPOSITORY_DIRNAME / BASE_DIRNAME
        base_dir.mkdir(parents=True, exist_ok=True)

        repository = cls(root)
        repository.save_config(config if config is not None else RepositoryConfig())

        return repository

    @classmethod
    def find(cls, path: Path | None = None) -> Repository:
        """Find the repository containing `path`, searching upwards."""
        candidate = (path if path is not None else Path.cwd()).resolve()

        for directory in [candidate, *candidate.parents]:
            if (directory / REPOSITORY_DIRNAME).is_dir():
                return cls(directory)

        raise PaprikaUserError(
            f"No paprika recipe directory found in {candidate} or any parent "
            "directory; run `paprika-recipes clone` to create one."
        )

    # -- Layout -------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def repository_dir(self) -> Path:
        return self._root / REPOSITORY_DIRNAME

    @property
    def base_dir(self) -> Path:
        return self.repository_dir / BASE_DIRNAME

    @property
    def attachments_dir(self) -> Path:
        return self._root / ATTACHMENTS_DIRNAME

    @property
    def photo_state_dir(self) -> Path:
        return self.repository_dir / PHOTOS_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.repository_dir / CONFIG_FILENAME

    @property
    def categories_path(self) -> Path:
        return self.repository_dir / CATEGORIES_FILENAME

    # -- Configuration ------------------------------------------------------

    @property
    def config(self) -> RepositoryConfig:
        if self._config is None:
            if self.config_path.is_file():
                with open(self.config_path, encoding="utf-8") as inf:
                    self._config = RepositoryConfig.from_dict(load_yaml(inf) or {})
            else:
                self._config = RepositoryConfig()

        return self._config

    def save_config(self, config: RepositoryConfig) -> None:
        with open(self.config_path, "w", encoding="utf-8") as outf:
            dump_yaml(config.as_dict(), outf)

        self._config = config
        self._format = None

    @property
    def format(self) -> DocumentFormat:
        """How this directory's files are spelled; see `DocumentFormat`."""
        if self._format is None:
            self._format = DocumentFormat(
                self.config.frontmatter_prefix,
                self.categories,
                self.photo_links(),
            )

        return self._format

    def photo_links(self) -> dict[str, str]:
        """Each recipe's thumbnail, mapped to the full-size file it links to.

        Only a full-size copy we actually wrote counts: a link must point at
        a file that is there, so a recipe whose gallery copy could not be
        downloaded shows its thumbnail without one.
        """
        aliases = {}

        for uid in self._photo_state_uids():
            state = self.read_photo_state(uid)

            if state is None or not state.photo:
                continue

            shown, _ = state.displayed()

            if shown != state.photo:
                aliases[state.photo] = shown

        return aliases

    @property
    def categories(self) -> CategoryNames:
        """The account's category names, as of the last sync."""
        return CategoryNames.load(self.categories_path)

    def update_categories(self, categories: CategoryNames) -> list[Path]:
        """Adopt a fresh list of category names; return the files rewritten.

        Recipes hold category uids, and only their files spell them as
        names, so a category renamed in the app changes nothing about any
        recipe -- but every file naming it is now spelled wrong.  See
        `respell` for how those files are put right.
        """
        old = self.format

        if old.categories == categories:
            return []

        categories.save(self.categories_path)
        self._format = None

        return self.respell(old)

    def respell(self, old: DocumentFormat) -> list[Path]:
        """Rewrite files written in `old` spelling; return the files rewritten.

        Some of what a file says is spelling rather than substance: category
        names stand for uids, and the photo embed shows a full-size file in
        place of the thumbnail the recipe names.  When either mapping moves
        -- a category renamed in the app, a gallery copy downloaded -- files
        written before the move are spelled wrong, and would read back as
        edits.  Each is read with the spelling it was written in and written
        back in the current one, so the recipe it holds, local edits and all,
        is exactly what it was.  A file whose spelling did not change is not
        touched.
        """
        new = self.format

        if new == old:
            return []

        rewritten = []

        for path in self.working_paths():
            with open(path, encoding="utf-8") as inf:
                content = inf.read()

            try:
                recipe, extras = old.parse(content, RemoteRecipe)
            except PaprikaUserError:
                # Whatever is wrong with it, the command about to read it will
                # say so; a change of spelling is no reason to touch it.
                continue

            respelled = new.render(recipe, extras)

            if respelled == old.render(recipe, extras):
                continue

            with open(path, "w", encoding="utf-8") as outf:
                outf.write(respelled)

            rewritten.append(path)

        return rewritten

    @property
    def ignore(self) -> IgnoreRules:
        """Which markdown files here are not recipes; see `ignore`.

        Read afresh every time rather than cached: the file is a few lines
        long, and a stale copy could let a recipe be written to a name that
        has since been excluded.
        """
        return IgnoreRules.load(self._root)

    # -- The base (pristine) copies -----------------------------------------

    def base_path_for(self, uid: str) -> Path:
        return self.base_dir / f"{uid}.json"

    def read_base(self, uid: str) -> RemoteRecipe | None:
        path = self.base_path_for(uid)

        if not path.is_file():
            return None

        with open(path, encoding="utf-8") as inf:
            # `store` normalises before writing, so this is a no-op for any
            # base copy written by this version.  It is here for the ones
            # written by 3.0.0, which kept Paprika's `\r`s and `null`s: without
            # it those recipes read as modified forever, and no pull could
            # settle them, because the server's copy has not moved.
            return normalize_recipe(RemoteRecipe.from_dict(json.load(inf)))

    def write_base(self, recipe: RemoteRecipe) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

        with open(self.base_path_for(recipe.uid), "w", encoding="utf-8") as outf:
            json.dump(recipe.as_dict(), outf, indent=2, sort_keys=True)

    def delete_base(self, uid: str) -> None:
        self.base_path_for(uid).unlink(missing_ok=True)

    def base_uids(self) -> set[str]:
        if not self.base_dir.is_dir():
            return set()

        return {path.stem for path in self.base_dir.glob("*.json")}

    # -- Photos ---------------------------------------------------------------

    def attachment_path(self, photo: str) -> Path:
        """Where a photo named `photo` lives.

        The name is the server's, so it is checked rather than trusted: a
        name that would land anywhere but directly inside `attachments/` is
        refused instead of written.
        """
        if not photo or "/" in photo or "\\" in photo or photo in (".", ".."):
            raise PaprikaUserError(
                f"{photo!r} is not a name we are willing to write a photo to."
            )

        return self.attachments_dir / photo

    def read_attachment(self, photo: str) -> bytes:
        """The bytes of a photo in `attachments/`, for sending to the server."""
        path = self.attachment_path(photo)

        if not path.is_file():
            raise PaprikaUserError(
                f"there is no {ATTACHMENTS_DIRNAME}/{photo} for its embed to "
                "point at"
            )

        return path.read_bytes()

    def read_photo_state(self, uid: str) -> PhotoState | None:
        path = self._photo_state_path(uid)

        if not path.is_file():
            return None

        with open(path, encoding="utf-8") as inf:
            try:
                data = json.load(inf)
            except ValueError:
                return None

        gallery = data.get("gallery") or {}

        return PhotoState(
            photo=str(data.get("photo", "")),
            photo_hash=str(data.get("photo_hash", "")),
            content_hash=str(data.get("content_hash", "")),
            large=str(data.get("large", "") or ""),
            gallery={
                str(name): GalleryPhotoState.from_dict(entry)
                for name, entry in gallery.items()
                if isinstance(entry, dict)
            },
        )

    def _save_photo_state(self, uid: str, state: PhotoState) -> None:
        path = self._photo_state_path(uid)

        if not state.photo and not state.gallery:
            path.unlink(missing_ok=True)
        else:
            self.photo_state_dir.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as outf:
                json.dump(asdict(state), outf, indent=2, sort_keys=True)

        # Where the embeds link to may have moved; see `photo_links`.
        self._format = None

    def write_photo(self, uid: str, photo: str, photo_hash: str, data: bytes) -> Path:
        """Write a recipe's thumbnail into `attachments/`, and remember it."""
        path = self.attachment_path(photo)
        state = self.read_photo_state(uid) or PhotoState()

        if state.photo and state.photo != photo and state.photo not in state.gallery:
            # The recipe's photo was replaced under a new name; the old file
            # is nobody's now, and leaving it would litter the vault.
            self.attachment_path(state.photo).unlink(missing_ok=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        self._save_photo_state(
            uid,
            replace(
                state,
                photo=photo,
                photo_hash=photo_hash,
                content_hash=hashlib.sha256(data).hexdigest(),
            ),
        )

        return path

    def drop_thumbnail(self, uid: str) -> None:
        """Remove a recipe's thumbnail, leaving its gallery alone."""
        state = self.read_photo_state(uid)

        if state is None:
            return

        if state.photo and state.photo not in state.gallery:
            self.attachment_path(state.photo).unlink(missing_ok=True)

        self._save_photo_state(
            uid, replace(state, photo="", photo_hash="", content_hash="", large="")
        )

    def write_gallery_photo(
        self, uid: str, filename: str, record: GalleryPhotoState, data: bytes
    ) -> Path:
        """Write one gallery photo into `attachments/`, and remember it."""
        path = self.attachment_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        state = self.read_photo_state(uid) or PhotoState()
        gallery = dict(state.gallery)
        gallery[filename] = replace(
            record, content_hash=hashlib.sha256(data).hexdigest()
        )
        self._save_photo_state(uid, replace(state, gallery=gallery))

        return path

    def record_gallery_photo(
        self, uid: str, filename: str, record: GalleryPhotoState
    ) -> None:
        """Update what we know of a gallery photo whose bytes did not change."""
        state = self.read_photo_state(uid) or PhotoState()
        gallery = dict(state.gallery)
        content_hash = gallery[filename].content_hash if filename in gallery else ""
        gallery[filename] = replace(record, content_hash=content_hash)
        self._save_photo_state(uid, replace(state, gallery=gallery))

    def drop_gallery_photo(self, uid: str, filename: str) -> None:
        """Remove one gallery photo from `attachments/`, and the record of it."""
        state = self.read_photo_state(uid)

        if state is None or filename not in state.gallery:
            return

        if filename != state.photo:
            self.attachment_path(filename).unlink(missing_ok=True)

        gallery = {k: v for k, v in state.gallery.items() if k != filename}
        large = state.large if state.large != filename else ""
        self._save_photo_state(uid, replace(state, gallery=gallery, large=large))

    def set_large_photo(self, uid: str, filename: str) -> None:
        """Say which gallery photo the embed should show; "" for the thumbnail."""
        state = self.read_photo_state(uid)

        if state is None or state.large == filename:
            return

        self._save_photo_state(uid, replace(state, large=filename))

    def drop_photo(self, uid: str) -> None:
        """Remove all of a recipe's photos from `attachments/`, and their record."""
        state = self.read_photo_state(uid)

        if state is not None:
            for filename in {state.photo, *state.gallery} - {""}:
                self.attachment_path(filename).unlink(missing_ok=True)

        self._photo_state_path(uid).unlink(missing_ok=True)
        self._format = None

    def mirrored_files(self) -> set[str]:
        """Every attachment that is a copy of something on the server."""
        files: set[str] = set()

        for uid in self._photo_state_uids():
            state = self.read_photo_state(uid)

            if state is not None:
                files |= {state.photo, *state.gallery} - {""}

        return files

    def _photo_state_path(self, uid: str) -> Path:
        return self.photo_state_dir / f"{uid}.json"

    def _photo_state_uids(self) -> list[str]:
        if not self.photo_state_dir.is_dir():
            return []

        return sorted(path.stem for path in self.photo_state_dir.glob("*.json"))

    def _differs(self, filename: str, content_hash: str) -> bool:
        """Do the bytes in `attachments/filename` differ from those we wrote?

        A missing file has not been replaced; it is `pull`'s to heal.
        """
        try:
            path = self.attachment_path(filename)
        except PaprikaUserError:
            return False

        if not path.is_file():
            return False

        return hashlib.sha256(path.read_bytes()).hexdigest() != content_hash

    def _photo_replaced(self, uid: str, recipe: RemoteRecipe) -> bool:
        """Has the user swapped out the image behind an unchanged embed?

        Only asked about an attachment we downloaded ourselves: when the
        embed names anything else, the *field* has changed and ordinary
        change detection already sees it.  The file checked is the thumbnail
        the embed shows; the full-size copy it links to is a gallery photo,
        whose replacement `_gallery_replaced` notices.  Comparing a digest of
        the file against the one recorded at download time is what Paprika's
        own `photo_hash` cannot do for us.
        """
        state = self.read_photo_state(uid)

        if state is None or not state.photo or state.photo != recipe.photo:
            return False

        return self._differs(state.photo, state.content_hash)

    def _gallery_replaced(self, uid: str, recipe: RemoteRecipe) -> frozenset[str]:
        """The gallery files the recipe lists whose bytes were swapped out."""
        state = self.read_photo_state(uid)

        if state is None:
            return frozenset()

        listed = {filename for filename, _ in recipe.gallery}

        return frozenset(
            filename
            for filename, record in state.gallery.items()
            if filename in listed and self._differs(filename, record.content_hash)
        )

    # -- The working directory ----------------------------------------------

    def working_paths(self) -> Iterator[Path]:
        yield from recipe_files(self._root, self.ignore)

    def read_working(self, path: Path) -> RemoteRecipe:
        return read_recipe(path, RemoteRecipe, self.format)

    def read_document(self, path: Path) -> tuple[RemoteRecipe, Extras]:
        """Read a working file as both a recipe and everything else it holds."""
        with open(path, encoding="utf-8") as inf:
            content = inf.read()

        try:
            return self.format.parse(content, RemoteRecipe)
        except PaprikaUserError as e:
            raise PaprikaUserError(f"{path}: {e}")

    def read_extras(self, path: Path) -> Extras:
        """The parts of a working file that are not the recipe; see `markdown`."""
        if not path.is_file():
            return Extras()

        with open(path, encoding="utf-8") as inf:
            return self.format.read_extras(inf.read(), RemoteRecipe)

    def write_working(
        self,
        recipe: RemoteRecipe,
        path: Path | None = None,
        extras: Extras | None = None,
    ) -> Path:
        lossy = self.format.find_lossy_fields(recipe, extras)
        if lossy:
            raise PaprikaUserError(
                f"Refusing to write {recipe.name!r}: {', '.join(lossy)} "
                "could not be read back from the markdown we would have "
                "written. Please report this as a bug."
            )

        if path is None:
            path = self.path_for(recipe)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as outf:
            outf.write(self.format.render(recipe, extras))

        return path

    def store(
        self,
        recipe: RemoteRecipe,
        existing: dict[str, Path] | None = None,
        base: RemoteRecipe | None = None,
    ) -> Path:
        """Record a recipe as pulled: write the working file and its base copy.

        This is the only way the two should ever be written together, because
        it normalises once and uses the result for both. Writing them
        separately risks a base copy that does not render to the file beside
        it, which would make the recipe look permanently modified.

        `base` says what the *server* holds, for the one case where that is
        not what we are writing to the file: after a merge the file holds the
        merged result, while the base copy has to hold the server's copy.
        Otherwise the merge would look like it had already been pushed, and
        the optimistic-lock check would refuse to push it.
        """
        normalized = normalize_recipe(recipe)
        path = self.path_for(normalized, existing)

        self.write_working(normalized, path, self.read_extras(path))
        self.write_base(normalize_recipe(base) if base is not None else normalized)

        return path

    def restore(self, uid: str) -> Path:
        """Put a recipe's working file back the way it last arrived.

        This is the undo for local edits, including deleting the file: the
        base copy is a complete recipe, so there is always something to
        restore to as long as the recipe has been pulled at least once.
        """
        base = self.read_base(uid)

        if base is None:
            raise PaprikaUserError(
                f"There is no record of having pulled the recipe {uid}, "
                "so there is nothing to restore it to."
            )

        state = self.read_photo_state(uid)

        if state is not None:
            # An image swapped out in place is a local change like any other,
            # so restoring discards it too.  We cannot re-download it from
            # here -- restore works without the network -- but removing the
            # file is enough: the state file still says which photo belongs
            # there, so the next pull sees it missing and puts it back.
            if self._photo_replaced(uid, base):
                self.attachment_path(state.photo).unlink(missing_ok=True)

            for filename in self._gallery_replaced(uid, base):
                self.attachment_path(filename).unlink(missing_ok=True)

        return self.store(base)

    def path_for(
        self, recipe: RemoteRecipe, existing: dict[str, Path] | None = None
    ) -> Path:
        """Choose a working file path for a recipe, avoiding collisions.

        Callers placing many recipes at once should pass `existing` (and keep
        it up to date themselves) to avoid re-reading the working directory
        for every single recipe.
        """
        if existing is None:
            existing = self.paths_by_uid()

        if recipe.uid in existing:
            return existing[recipe.uid]

        return unique_path(
            self._root,
            recipe,
            {path.resolve() for path in existing.values()},
            self.ignore,
        )

    def scan(self) -> tuple[dict[str, Path], list[Path]]:
        """Every recipe file, split by whether it says which recipe it is.

        We read the uid out of each file rather than keeping an index, so that
        renaming a file in the working directory just works.  A file with no
        uid is not an error and not ignored: it is a recipe someone wrote
        that has not been to Paprika yet.
        """
        tracked: dict[str, Path] = {}
        untracked: list[Path] = []

        for path in self.working_paths():
            recipe = self.read_working(path)

            if not recipe.uid:
                untracked.append(path)
                continue

            if recipe.uid in tracked:
                raise PaprikaUserError(
                    f"Two files claim the same recipe uid {recipe.uid}: "
                    f"{tracked[recipe.uid]} and {path}."
                )

            tracked[recipe.uid] = path

        return tracked, untracked

    def paths_by_uid(self) -> dict[str, Path]:
        """Map each recipe's uid to the file holding it."""
        return self.scan()[0]

    def adopt(self, entry: WorkingRecipe) -> WorkingRecipe:
        """Give an untracked recipe a uid of its own, and write it into its file.

        Done at the moment the recipe is first sent to Paprika, rather than
        when its file is first read.  Two reasons, and the second is the one
        that matters:

        `status` is what someone runs to find out what is *about* to happen,
        and a command asked that question should not answer it by writing to
        their files.

        More importantly, assigning on read would make an unreadable directory
        unrecoverable.  If every file suddenly stopped being identifiable --
        a `frontmatter_prefix` changed, a vault plugin that prunes frontmatter
        keys -- then a `status` run to find out what on earth had happened
        would rewrite all of them, and the original uids would survive only as
        leftover frontmatter.  Writing nothing until a push leaves the
        evidence intact, which is what lets `Syncer` refuse the whole thing.
        """
        if entry.recipe is None or entry.path is None:
            raise PaprikaProgrammingError(
                f"{entry.name} has no file to write a uid into."
            )

        recipe = replace(entry.recipe, uid=str(uuid.uuid4()).upper())
        self.write_working(recipe, entry.path, entry.extras)

        return replace(entry, uid=recipe.uid, recipe=recipe)

    def _refuse_ignored_recipes(self, missing: set[str]) -> None:
        """Stop before an ignore rule makes a tracked recipe look deleted.

        A recipe whose file is matched by `.paprikaignore` is invisible to
        `scan`, so from in here it looks exactly like a file the user deleted
        -- and `push` would move it into Paprika's trash.  Adding an ignore
        pattern should never do that, so an ignored file that carries the
        uid of a recipe we are missing is refused instead.
        """
        if not missing:
            return

        ignore = self.ignore

        for path in markdown_files(self._root):
            if not ignore.ignores(path.relative_to(self._root)):
                continue

            try:
                uid = self.read_working(path).uid
            except PaprikaUserError:
                # Not something we could have written; plainly not a recipe.
                continue

            if uid in missing:
                raise PaprikaUserError(
                    f"{path} is the recipe {uid}, but {IGNORE_FILENAME} "
                    "excludes it, which would make it look deleted. Move the "
                    "file somewhere that is not ignored, or change the "
                    "pattern that matches it."
                )

    # -- Change detection ---------------------------------------------------

    def is_modified(self, uid: str, path: Path) -> bool:
        """Has the working file diverged from the recipe we last pulled?

        Compares rendered text rather than parsed fields; see the module
        docstring for why that distinction is the load-bearing one.
        """
        base = self.read_base(uid)

        if base is None:
            return True

        with open(path, encoding="utf-8") as inf:
            content = inf.read()

        # Render the base with whatever of the file is the user's own, so
        # that their `tags:` or their own `## ` section is not mistaken for an
        # edit to the recipe.
        return documents_differ(
            content,
            self.format.render(base, self.format.read_extras(content, RemoteRecipe)),
        )

    def status(self) -> list[WorkingRecipe]:
        paths, untracked = self.scan()
        base_uids = self.base_uids()

        self._refuse_ignored_recipes(base_uids - set(paths))

        result: list[WorkingRecipe] = []

        for untracked_path in untracked:
            recipe, extras = self.read_document(untracked_path)

            result.append(
                WorkingRecipe(
                    uid="",
                    status=Status.ADDED,
                    path=untracked_path,
                    recipe=recipe,
                    base=None,
                    extras=extras,
                )
            )

        for uid in sorted(set(paths) | base_uids):
            path = paths.get(uid)
            base = self.read_base(uid)

            if path is None:
                result.append(
                    WorkingRecipe(
                        uid=uid,
                        status=Status.DELETED,
                        path=None,
                        recipe=None,
                        base=base,
                    )
                )
                continue

            recipe, extras = self.read_document(path)
            photo_replaced = self._photo_replaced(uid, recipe)
            gallery_replaced = self._gallery_replaced(uid, recipe)

            if uid not in base_uids:
                status = Status.ADDED
            elif photo_replaced or gallery_replaced or self.is_modified(uid, path):
                # A replaced image is a modification the file's text cannot
                # show, so it is asked about before the text is.
                status = Status.MODIFIED
            else:
                status = Status.UNCHANGED

            result.append(
                WorkingRecipe(
                    uid=uid,
                    status=status,
                    path=path,
                    recipe=recipe,
                    base=base,
                    extras=extras,
                    photo_replaced=photo_replaced,
                    gallery_replaced=gallery_replaced,
                )
            )

        return result


def recipe_files(root: Path, ignore: IgnoreRules | None = None) -> Iterator[Path]:
    """Every recipe file under `root`, skipping our bookkeeping and ignored files.

    `ignore` defaults to whatever `root`'s own `.paprikaignore` says.
    """
    for path in markdown_files(root):
        if not _ignored(root, path, ignore):
            yield path


def markdown_files(root: Path) -> Iterator[Path]:
    """Every markdown file under `root` except our own bookkeeping."""
    for path in sorted(root.rglob(f"*{RECIPE_SUFFIX}")):
        if REPOSITORY_DIRNAME in path.relative_to(root).parts:
            continue

        yield path


def _ignored(root: Path, path: Path, ignore: IgnoreRules | None = None) -> bool:
    if ignore is None:
        ignore = IgnoreRules.load(root)

    return ignore.ignores(path.relative_to(root))


def read_recipe(
    path: Path, recipe_class: type[T], document_format: DocumentFormat | None = None
) -> T:
    """Read a recipe from a file, blaming the file if it cannot be read."""
    with open(path, encoding="utf-8") as inf:
        content = inf.read()

    try:
        if document_format is None:
            return parse_recipe(content, recipe_class)

        return document_format.parse_recipe(content, recipe_class)
    except PaprikaUserError as e:
        raise PaprikaUserError(f"{path}: {e}")


def unique_path(
    root: Path,
    recipe: BaseRecipe,
    taken: Container[Path],
    ignore: IgnoreRules | None = None,
) -> Path:
    """Choose a file for a recipe, avoiding names already spoken for.

    A name the directory's `.paprikaignore` excludes is spoken for too: a
    recipe written there would be invisible to the next `status`, which would
    then report it deleted and have `push` move it into Paprika's trash.
    """
    if ignore is None:
        ignore = IgnoreRules.load(root)

    # A recipe with neither a name nor a uid has nothing to be called; that
    # only happens for a file someone wrote and left blank, which is theirs
    # to name rather than ours.
    stem = safe_filename(recipe.name) or recipe.uid or "Untitled"

    # Two different recipes can share a name; fall back to disambiguating
    # with a slice of the uid, which is unique by construction.
    suffix = recipe.uid.split("-")[0] or uuid.uuid4().hex[:8]

    candidates = [
        root / f"{stem}{RECIPE_SUFFIX}",
        root / f"{stem} ({suffix}){RECIPE_SUFFIX}",
        root / f"{recipe.uid or uuid.uuid4().hex}{RECIPE_SUFFIX}",
    ]

    for candidate in candidates:
        if ignore.ignores(candidate.relative_to(root)):
            continue

        if candidate.resolve() not in taken and not candidate.exists():
            return candidate

    # Everything is taken; the disambiguated name is what we always fell
    # back to before there were ignore rules, so it stays the first choice.
    for candidate in (*candidates[1:], candidates[0]):
        if not ignore.ignores(candidate.relative_to(root)):
            return candidate

    raise PaprikaUserError(
        f"Every file name we could give {recipe.name or recipe.uid!r} is "
        f"excluded by {root / IGNORE_FILENAME}."
    )


def safe_filename(name: str) -> str:
    """Make a recipe's name safe to use as a filename."""
    cleaned = "".join(
        character
        for character in name
        if character not in UNSAFE_FILENAME_CHARACTERS and character.isprintable()
    )

    # A leading dot would hide the file; a trailing dot or space is invalid on
    # Windows and confusing everywhere else.
    return cleaned.strip(". ")
