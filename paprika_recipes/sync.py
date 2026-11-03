"""Reconciling a working directory against a Paprika account.

Syncing here means what it means in git: each side may have moved since we last
looked, and our job is to work out which, apply whatever can be applied safely,
and say plainly what could not.

Three questions decide every recipe, and they are asked in this order:

1.  *Has the server's copy moved?*  The sync index gives us a `(uid, hash)`
    pair for every recipe in a single request, and the base copy remembers the
    hash the recipe carried when we pulled it.  If they agree, the server's
    copy is where we left it.  (This is the only thing Paprika's hash is good
    for -- it folds in the photo, so we cannot compute it ourselves.)
2.  *Has the local copy moved?*  `WorkingRecipe.has_local_changes` answers
    this by diffing the file against the base copy, ignoring edits that change
    only how the file is written rather than what it says.
3.  *Did both move?*  Then we merge.  The base copy is exactly the third
    input a three-way merge needs, so edits to different parts of a recipe
    both survive and only genuinely overlapping ones need anyone's attention.
    See `merge` for what can and cannot be reconciled that way.

A merged file is left holding the merge while its base copy holds the
server's copy, which is what makes the result a local change like any other:
`status` shows it, `push` sends it, `restore` throws it away.  A merge that
needed conflict markers is not pushable until the markers are gone -- `push`
refuses it -- so a half-resolved merge cannot reach the server.

Deletion travels in both directions, but never destructively.  A recipe that
vanishes from the server is removed from the working directory; a file deleted
locally is pushed as a move into Paprika's trash.  Paprika's sync API has no
delete verb at all -- only an `in_trash` flag -- which happens to be exactly
the semantics worth wanting: the recipe is recoverable from the app's own
trash, and locally it can be brought back with `restore`, since the base copy
is a complete recipe and is not discarded until the trashing succeeds.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from .categories import CategoryNames
from .exceptions import PaprikaError, PaprikaUserError
from .images import PreparedPhoto, prepare_photo
from .markdown import (
    ATTACHMENTS_DIRNAME,
    GALLERY_FIELD,
    PHOTO_FIELD,
    PHOTO_FIELDS,
    foreign_uid_key,
    normalize_gallery,
)
from .merge import Merge, has_conflict_markers, merge_recipes
from .remote import RemotePhoto, RemoteRecipe
from .repository import (
    CONFIG_FILENAME,
    REPOSITORY_DIRNAME,
    GalleryPhotoState,
    PhotoState,
    Repository,
    Status,
    WorkingRecipe,
)


class RemoteAccount(Protocol):
    """The part of `Remote` that syncing needs; a seam for testing."""

    def get_recipe_index(self) -> dict[str, str]: ...

    def get_categories(self) -> CategoryNames: ...

    def get_recipe_by_id(self, id: str, hash: str) -> RemoteRecipe: ...

    def download_photo(self, url: str) -> bytes: ...

    def photo_download_url(self, uid: str) -> str: ...

    def gallery_photo_url(self, uid: str) -> str: ...

    def upload_recipe(
        self, recipe: RemoteRecipe, photo_upload: bytes | None = None
    ) -> RemoteRecipe: ...

    def upload_photo(self, photo: RemotePhoto, image: bytes | None = None) -> None: ...

    def get_photos(self) -> list[RemotePhoto]: ...

    def notify(self) -> None: ...


class Action(Enum):
    """What became of a single recipe during a sync."""

    #: Written to the working directory for the first time.
    ADDED = "added"
    #: Rewritten in the working directory from a newer copy on the server.
    UPDATED = "updated"
    #: Changed on both sides and reconciled without anyone having to help.
    MERGED = "merged"
    #: Removed from the working directory; it is no longer on the server.
    REMOVED = "removed"
    #: Uploaded to the server as a recipe it did not have.
    CREATED = "created"
    #: Uploaded to the server, replacing the copy it had.
    UPLOADED = "uploaded"
    #: Moved into Paprika's trash, because its file was deleted locally.
    TRASHED = "trashed"
    #: Put back the way it last arrived from the server.
    RESTORED = "restored"
    #: Changed on both sides; left alone for the user to resolve.
    CONFLICT = "conflict"
    #: Nothing to do, for a reason worth mentioning.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Change:
    action: Action
    uid: str
    name: str
    detail: str = ""


@dataclass
class SyncReport:
    """What a sync did, and what it declined to do."""

    changes: list[Change] = field(default_factory=list)
    unchanged: int = 0

    def record(self, action: Action, uid: str, name: str, detail: str = "") -> None:
        self.changes.append(Change(action, uid, name or uid, detail))

    def of(self, *actions: Action) -> list[Change]:
        """The changes matching any of `actions`, in the order they happened."""
        return [change for change in self.changes if change.action in actions]

    @property
    def conflicts(self) -> list[Change]:
        return self.of(Action.CONFLICT)

    def __bool__(self) -> bool:
        """Whether anything happened at all."""
        return bool(self.changes)


def refuse_unidentifiable(entries: list[WorkingRecipe]) -> None:
    """Stop a push that has clearly misread the directory rather than acting on it.

    Every recipe file carries the uid of the recipe it is.  If we stop being
    able to read those -- a `frontmatter_prefix` that no longer matches what
    the files were written with, a vault plugin that prunes frontmatter keys
    it does not recognise -- then the directory looks, from in here, exactly
    like someone having deleted every recipe they had and written the same
    number of new ones.  Acting on that reading would trash the whole account
    and upload a duplicate of everything in it.

    Two signals distinguish that from the real thing, because in the real
    thing the *files* are gone:

    The precise one is a file with no uid that nonetheless carries something
    called a uid.  A recipe somebody wrote themselves has no such key; a
    recipe we wrote and can no longer read does.

    The blunt one is for a directory whose uids were removed outright, which
    leaves no evidence to find: every recipe we have ever pulled is missing
    its file, and there are unidentified files sitting there instead.  One
    recipe deleted and one written is an ordinary afternoon, so this needs at
    least two, and even then it is only ever raised in preference to deleting
    somebody's entire collection.
    """
    untracked = [entry for entry in entries if entry.untracked]

    if not untracked:
        return

    for entry in untracked:
        key = foreign_uid_key(entry.extras)

        if key:
            raise PaprikaUserError(
                f"{entry.path} has a `{key}:` in its frontmatter, but no uid "
                "this directory can read. That usually means the file was "
                "written with a different `frontmatter_prefix` than the one "
                f"in {REPOSITORY_DIRNAME}/{CONFIG_FILENAME}. Nothing has been "
                "pushed; a push would upload this as a new recipe and trash "
                "the one it already is."
            )

    tracked = [entry for entry in entries if not entry.untracked]
    missing = [entry for entry in tracked if entry.status is Status.DELETED]

    if len(missing) > 1 and len(missing) == len(tracked):
        raise PaprikaUserError(
            f"Every one of the {len(missing)} recipes this directory has "
            f"pulled is missing its file, and {len(untracked)} "
            f"{'file' if len(untracked) == 1 else 'files'} we cannot identify "
            "are here instead. That is what a directory looks like when its "
            "recipe files have stopped being readable, so nothing has been "
            "pushed.\n\nIf you really did mean to delete every recipe, move "
            "the unidentified files out of the directory, push, and move them "
            "back."
        )


def restore(repository: Repository, entries: Iterable[WorkingRecipe]) -> SyncReport:
    """Put the given recipes back the way they last arrived from the server.

    This needs no network at all: the base copies are the only thing being
    read.  It is the undo for anything `status` reports -- an edit, or the
    deletion of a file -- but pointedly not for a recipe that was never
    pulled, since there is nothing to put such a file back to and deleting
    someone's new work is not an undo.
    """
    report = SyncReport()

    for entry in entries:
        if entry.status is Status.UNCHANGED:
            report.unchanged += 1
            continue

        if entry.base is None:
            report.record(
                Action.SKIPPED,
                entry.uid,
                entry.name,
                "was never pulled from Paprika, so there is nothing to "
                "restore it to; delete the file yourself if you meant to",
            )
            continue

        repository.restore(entry.uid)
        report.record(Action.RESTORED, entry.uid, entry.name)

    return report


class Syncer:
    """Reconciles a `Repository` against a Paprika account."""

    def __init__(
        self,
        repository: Repository,
        remote: RemoteAccount,
        on_recipe: Callable[[str], None] | None = None,
    ):
        self._repository = repository
        self._remote = remote
        self._on_recipe = on_recipe

    # -- Pulling ------------------------------------------------------------

    def pull(self, dry_run: bool = False) -> SyncReport:
        """Bring the working directory up to date with the server."""
        report = SyncReport()

        self._refresh_categories(dry_run)
        spelling = self._repository.format

        index = self._remote.get_recipe_index()
        galleries = self._galleries()
        entries = {entry.uid: entry for entry in self._repository.status()}
        paths = {
            uid: entry.path for uid, entry in entries.items() if entry.path is not None
        }

        # Recipes the server still knows about but has moved to the trash; as
        # far as a recipe directory is concerned those are gone.
        trashed: set[str] = set()

        for uid in sorted(index):
            entry = entries.get(uid)
            gallery = _projection(galleries.get(uid, []))

            if (
                entry is not None
                and entry.base is not None
                and entry.base.hash == index[uid]
            ):
                if entry.base.gallery == gallery:
                    # The server's copy has not moved since we pulled it, so
                    # there is nothing to bring down -- whatever the user has
                    # done to the file locally is their business until they
                    # push it.  Missing photos are healed afterwards; see
                    # `_sync_photos`.
                    report.unchanged += 1
                    continue

                # Only the gallery moved, which leaves the recipe's own hash
                # alone; the recipe itself is the one we already have.
                recipe = entry.base
            else:
                recipe = self._fetch(uid, index[uid])

            if recipe.in_trash:
                trashed.add(uid)
                continue

            recipe = replace(recipe, gallery=gallery)

            if entry is None:
                if not dry_run:
                    paths[uid] = self._repository.store(recipe, paths)
                report.record(Action.ADDED, uid, recipe.name)
            elif entry.status is Status.ADDED:
                report.record(
                    Action.CONFLICT,
                    uid,
                    recipe.name,
                    "is on the server, but we have no record of having pulled it",
                )
            elif entry.status is Status.DELETED:
                report.record(
                    Action.CONFLICT,
                    uid,
                    recipe.name,
                    "was deleted locally, but has changed on the server; "
                    "push to trash it anyway, or `restore` to keep it",
                )
            elif entry.has_local_changes():
                paths[uid] = self._pull_merge(report, entry, recipe, paths, dry_run)
            else:
                if not dry_run:
                    paths[uid] = self._repository.store(recipe, paths)
                report.record(Action.UPDATED, uid, recipe.name)

        self._pull_removals(report, set(index) - trashed, entries, dry_run)

        if not dry_run:
            # Photos come last, once every base copy says which ones belong
            # where; the files are then respelled to link the photos that
            # just arrived.  See `Repository.respell`.
            self._sync_photos(report, galleries, self._repository.base_uids())
            self._repository.respell(spelling)

        return report

    def _pull_merge(
        self,
        report: SyncReport,
        entry: WorkingRecipe,
        remote: RemoteRecipe,
        paths: dict[str, Path],
        dry_run: bool,
    ) -> Path:
        """Reconcile a recipe that has moved on both sides."""
        # Guaranteed by the caller: this path is only reached for a recipe
        # that is present, tracked and locally changed.
        assert entry.base is not None
        assert entry.recipe is not None
        assert entry.path is not None

        merge = merge_recipes(entry.base, entry.recipe, remote)

        # Images swapped out in place are local changes that field-level
        # merging cannot see -- the file's text reads unchanged -- so a
        # collision with a server-side change to the same thing is refused
        # here instead.
        if merge.merged and entry.photo_replaced and remote.photo != entry.base.photo:
            merge = Merge(None, merge.conflicted, (*merge.unmergeable, PHOTO_FIELD))

        if (
            merge.merged
            and entry.gallery_replaced
            and remote.gallery != entry.base.gallery
        ):
            merge = Merge(None, merge.conflicted, (*merge.unmergeable, GALLERY_FIELD))

        if not merge.merged:
            report.record(
                Action.CONFLICT,
                entry.uid,
                remote.name,
                f"changed on the server and locally, and {_and(merge.unmergeable)} "
                "cannot hold both answers; nothing was changed",
            )
            return entry.path

        if not dry_run:
            # The file holds the merge; the base copy holds what the server
            # holds, so that the merge is a local change we can then push.
            paths[entry.uid] = self._repository.store(merge.recipe, paths, base=remote)

        if merge.clean:
            report.record(
                Action.MERGED,
                entry.uid,
                remote.name,
                "changed on both sides; your edits were kept",
            )
        else:
            report.record(
                Action.CONFLICT,
                entry.uid,
                remote.name,
                f"{_and(merge.conflicted)} changed on both sides; resolve the "
                "conflict markers left in the file, then push",
            )

        return paths[entry.uid]

    def _pull_removals(
        self,
        report: SyncReport,
        live_uids: set[str],
        entries: dict[str, WorkingRecipe],
        dry_run: bool,
    ) -> None:
        """Drop recipes the server no longer has."""
        for uid in sorted(self._repository.base_uids() - live_uids):
            entry = entries[uid]
            name = entry.base.name if entry.base is not None else uid

            if entry.status is not Status.DELETED and entry.has_local_changes():
                report.record(
                    Action.CONFLICT,
                    uid,
                    name,
                    "has changed locally, but is no longer on the server",
                )
                continue

            if not dry_run:
                if entry.path is not None:
                    entry.path.unlink(missing_ok=True)
                self._repository.delete_base(uid)
                self._repository.drop_photo(uid)

            report.record(Action.REMOVED, uid, name)

    # -- Pushing ------------------------------------------------------------

    def push(self, dry_run: bool = False) -> SyncReport:
        """Send local changes to the server."""
        report = SyncReport()

        # A category created in the app since the last pull is one the user
        # may already have typed into a file; it has to be known by name
        # before the files are read.
        categories = self._refresh_categories(dry_run)
        spelling = self._repository.format

        entries = self._repository.status()
        refuse_unidentifiable(entries)

        index = self._remote.get_recipe_index()
        galleries = self._galleries()
        uploaded: list[str] = []
        sources: set[str] = set()

        for entry in entries:
            uid = self._push_one(
                report, entry, index, galleries, dry_run, categories, sources
            )

            if uid:
                uploaded.append(uid)

        if dry_run:
            return report

        if report.of(Action.CREATED, Action.UPLOADED, Action.TRASHED):
            # Ask Paprika to tell its apps that something changed.
            self._remote.notify()

        if uploaded:
            # Then take the server's word for what each recipe now says, and
            # mirror the photos it now holds.
            self._refresh(report, uploaded)

        self._repository.respell(spelling)
        self._clean_up(report, sources)

        return report

    def _push_one(
        self,
        report: SyncReport,
        entry: WorkingRecipe,
        index: dict[str, str],
        galleries: dict[str, list[RemotePhoto]],
        dry_run: bool,
        categories: CategoryNames | None = None,
        sources: set[str] | None = None,
    ) -> str:
        """Push a single recipe; return its uid if it was sent to the server."""
        if entry.status is Status.UNCHANGED:
            report.unchanged += 1
            return ""

        if entry.status is Status.DELETED or entry.recipe is None:
            return self._push_removal(report, entry, index, dry_run)

        recipe = entry.recipe
        name = recipe.name

        if has_conflict_markers(recipe):
            report.record(
                Action.CONFLICT,
                entry.uid,
                name,
                "still has unresolved conflict markers in it; edit them out "
                "(or `restore` it) before pushing",
            )
            return ""

        unknown = _unknown_categories(entry, categories)

        if unknown:
            report.record(
                Action.SKIPPED,
                entry.uid,
                name,
                f"was not pushed: Paprika has no {_categories_named(unknown)}; "
                "create it in the app (or fix the spelling), then push again",
            )
            return ""

        if entry.status is Status.ADDED:
            if entry.untracked:
                if dry_run:
                    report.record(Action.CREATED, "", name)
                    return ""

                # Now that this recipe is about to exist on the server, it
                # needs an identity that outlives this run.
                entry = self._repository.adopt(entry)

                # Adopting only ever rewrites the uid of the recipe we just
                # read out of the file.
                assert entry.recipe is not None
                recipe = entry.recipe
            elif entry.uid in index:
                report.record(
                    Action.CONFLICT,
                    entry.uid,
                    name,
                    "is already on the server, but we have no record of having "
                    "pulled it; pull first to see what it says",
                )
                return ""
        elif not entry.has_local_changes():
            report.record(
                Action.SKIPPED,
                entry.uid,
                name,
                "differs only in how the file is written, not in what it says",
            )
            return ""
        elif entry.base is not None and (
            index.get(entry.uid) != entry.base.hash
            or _projection(galleries.get(entry.uid, [])) != entry.base.gallery
        ):
            report.record(
                Action.CONFLICT,
                entry.uid,
                name,
                "has changed on the server since it was pulled; pull first",
            )
            return ""

        action = Action.CREATED if entry.status is Status.ADDED else Action.UPLOADED

        try:
            gallery = self._plan_gallery(entry, recipe)
            plan = self._plan_photo(entry, recipe, gallery)
        except PaprikaUserError as e:
            report.record(Action.SKIPPED, entry.uid, name, f"was not pushed: {e}")
            return ""

        # A change to nothing but the gallery leaves the recipe itself alone.
        recipe_changed = entry.status is Status.ADDED or bool(
            set(entry.changed_fields()) - {GALLERY_FIELD}
        )

        if not dry_run:
            self._notify(name)

            for photo, image in gallery.uploads:
                self._remote.upload_photo(photo, image)

            if plan.gallery is not None:
                self._remote.upload_photo(*plan.gallery)

            if recipe_changed:
                self._remote.upload_recipe(plan.upload, photo_upload=plan.thumbnail)

            for photo in gallery.deletions:
                self._remote.upload_photo(photo)

            if sources is not None:
                sources |= gallery.sources | ({plan.source} - {""})

        # Uploading is the moment to say what else moved, or pointedly did
        # not: the rest of the file stays behind, and someone who wrote it
        # there deserves to be told rather than left to discover it.
        notes = []
        kept = entry.extras.describe()

        if kept:
            notes.append(f"{kept} stayed in your file")
        for note in (plan.note, gallery.note):
            if note:
                notes.append(note)

        report.record(action, entry.uid, name, "; ".join(notes))

        return "" if dry_run else entry.uid

    def _push_removal(
        self,
        report: SyncReport,
        entry: WorkingRecipe,
        index: dict[str, str],
        dry_run: bool,
    ) -> str:
        """Move a recipe whose file was deleted into Paprika's trash.

        The base copy is kept until the trashing has actually happened, so
        that a failure here leaves the recipe restorable rather than merely
        gone from both places.
        """
        name = entry.name

        if entry.base is None or entry.uid not in index:
            if not dry_run:
                self._repository.delete_base(entry.uid)
                self._repository.drop_photo(entry.uid)

            report.record(
                Action.REMOVED, entry.uid, name, "was already gone from the server"
            )
            return ""

        if index[entry.uid] != entry.base.hash:
            report.record(
                Action.CONFLICT,
                entry.uid,
                name,
                "was deleted locally, but has since changed on the server; "
                "pull to see what changed, or `restore` to keep it",
            )
            return ""

        if not dry_run:
            self._notify(name)
            self._remote.upload_recipe(replace(entry.base, in_trash=True))
            self._repository.delete_base(entry.uid)
            self._repository.drop_photo(entry.uid)

        report.record(Action.TRASHED, entry.uid, name)

        # Nothing to re-fetch: refreshing would write the file back out.
        return ""

    def _refresh(self, report: SyncReport, uids: Iterable[str]) -> None:
        """Re-pull recipes we just uploaded, so their base copies are the server's.

        Uploading rewrites a recipe's hash, and Paprika may normalise other
        parts of it besides; asking for the recipe back is the only way to know
        what the server now believes, and it leaves the working file matching
        its base copy rather than looking eternally modified.  The photos are
        then downloaded from the server too, rather than kept from what we
        sent, so that `attachments/` holds exactly the files Paprika holds.
        """
        index = self._remote.get_recipe_index()
        galleries = self._galleries()
        paths = self._repository.paths_by_uid()
        live = [uid for uid in uids if uid in index]

        for uid in live:
            recipe = self._fetch(uid, index[uid])
            recipe = replace(recipe, gallery=_projection(galleries.get(uid, [])))
            self._repository.store(recipe, paths)

        self._sync_photos(report, galleries, live)

    def _clean_up(self, report: SyncReport, sources: set[str]) -> None:
        """Remove the images a push uploaded from where the user put them.

        Each went up under a server-chosen name and came back down under it,
        so the file the user dropped into `attachments/` is now a duplicate.
        It is kept if anything still refers to it -- another recipe's file,
        or a download that failed and left nothing in its place.
        """
        if not sources or report.of(Action.SKIPPED):
            return

        referenced = self._repository.mirrored_files()

        for entry in self._repository.status():
            if entry.recipe is not None:
                referenced.add(entry.recipe.photo)
                referenced |= {filename for filename, _ in entry.recipe.gallery}

        for filename in sources - referenced:
            self._repository.attachment_path(filename).unlink(missing_ok=True)

    # -- Photos ---------------------------------------------------------------

    def _galleries(self) -> dict[str, list[RemotePhoto]]:
        """Every recipe's gallery photos, in order, from one request."""
        by_recipe: dict[str, list[RemotePhoto]] = defaultdict(list)

        for photo in self._remote.get_photos():
            if photo.deleted or not photo.filename or not photo.recipe_uid:
                continue

            by_recipe[photo.recipe_uid].append(photo)

        return {
            uid: sorted(photos, key=lambda p: (p.order_flag or 0, p.name, p.uid))
            for uid, photos in by_recipe.items()
        }

    def _prepare(self, filename: str) -> PreparedPhoto:
        """Read an image from `attachments/` and cut it the way Paprika wants.

        Raises `PaprikaUserError` when it is missing or not an image, so that
        the recipe can be reported rather than half-pushed.
        """
        data = self._repository.read_attachment(filename)

        try:
            return prepare_photo(data)
        except PaprikaUserError as e:
            raise PaprikaUserError(f"{ATTACHMENTS_DIRNAME}/{filename} {e}")

    def _plan_gallery(self, entry: WorkingRecipe, recipe: RemoteRecipe) -> _GalleryPlan:
        """Decide what a push means for one recipe's gallery.

        The `## Photos` list in the file is the whole interface.  A line
        naming a photo we mirrored keeps that photo, re-captioned or moved
        if its caption or position changed; any other line is a new photo,
        uploaded from `attachments/`; and a mirrored photo the list no longer
        names is taken down.  A mirrored file whose bytes were swapped out
        goes up as a new photo in its old one's place.
        """
        state = self._repository.read_photo_state(recipe.uid) if recipe.uid else None
        known = state.gallery if state is not None else {}
        base = entry.base.gallery if entry.base is not None else []
        local = recipe.gallery
        replaced = entry.gallery_replaced

        if local == base and not replaced:
            return _GalleryPlan()

        uploads: list[tuple[RemotePhoto, bytes | None]] = []
        renames: dict[str, str] = {}
        sources: set[str] = set()
        listed: set[str] = set()
        changed = 0

        for position, (filename, caption) in enumerate(local):
            listed.add(filename)
            record = known.get(filename)

            if record is not None and filename not in replaced:
                if record.name != caption or record.order_flag != position:
                    uploads.append(
                        (
                            RemotePhoto(
                                uid=record.uid,
                                recipe_uid=recipe.uid,
                                filename=filename,
                                name=caption,
                                order_flag=position,
                                hash=record.hash,
                            ),
                            None,
                        )
                    )
                    changed += 1
                continue

            prepared = self._prepare(filename)
            new = _photo_filename()
            uploads.append(
                (
                    RemotePhoto(
                        uid=new.removesuffix(".jpg"),
                        recipe_uid=recipe.uid,
                        filename=new,
                        name=caption,
                        order_flag=position,
                        hash=hashlib.sha256(prepared.full).hexdigest().upper(),
                    ),
                    prepared.full,
                )
            )
            renames[filename] = new

            if record is None:
                sources.add(filename)

        deletions = tuple(
            RemotePhoto(
                uid=record.uid,
                recipe_uid=recipe.uid,
                filename=filename,
                name=record.name,
                order_flag=record.order_flag,
                hash=record.hash,
                deleted=True,
            )
            for filename, record in sorted(known.items())
            if filename not in listed or filename in replaced
        )

        parts = []
        added = len(renames)
        removed = len(deletions) - len(replaced & listed)

        if added:
            parts.append(f"{added} photo{'s' if added != 1 else ''} added")
        if removed:
            parts.append(f"{removed} removed")
        if changed:
            parts.append(f"{changed} re-captioned or moved")

        return _GalleryPlan(
            uploads=tuple(uploads),
            deletions=deletions,
            renames=renames,
            sources=frozenset(sources),
            note=f"its gallery changed ({', '.join(parts)})" if parts else "",
        )

    def _plan_photo(
        self, entry: WorkingRecipe, recipe: RemoteRecipe, gallery: _GalleryPlan
    ) -> _PhotoPlan:
        """Decide what a push means for one recipe's own photo.

        The embed beneath the title is the whole interface: an embed matching
        the base copy means the photo is not what is being pushed, a missing
        one means the photo is being removed, and any other embed -- or the
        same embed over replaced bytes -- means a photo is going up.  A photo
        going up becomes a thumbnail on the recipe and a full-size copy in
        the gallery, exactly as the app would store it; if the embed names a
        picture that is already in the gallery, or about to be, that copy is
        reused rather than uploaded twice.

        Raises `PaprikaUserError` when the embed points at something that is
        not there or not an image.
        """
        server = entry.base if entry.base is not None else RemoteRecipe(uid=recipe.uid)
        deleted = {photo.filename for photo in gallery.deletions}

        if recipe.photo == server.photo and not entry.photo_replaced:
            # Not the photo's turn: ship the server's own photo fields back,
            # pointing the full-size link at wherever the gallery put it.
            fields = {name: getattr(server, name) for name in PHOTO_FIELDS}
            large = fields["photo_large"]

            if large in gallery.renames:
                fields["photo_large"] = gallery.renames[large]
            elif large in deleted:
                fields["photo_large"] = None

            return _PhotoPlan(upload=replace(recipe, **fields))

        if not recipe.photo:
            return _PhotoPlan(
                upload=replace(
                    recipe, photo="", photo_hash="", photo_large=None, photo_url=None
                ),
                note="its photo was removed from Paprika too",
            )

        state = self._repository.read_photo_state(recipe.uid) if recipe.uid else None
        known = state.gallery if state is not None else {}
        prepared = self._prepare(recipe.photo)
        extra: tuple[RemotePhoto, bytes] | None = None
        source = ""

        if recipe.photo in gallery.renames:
            large = gallery.renames[recipe.photo]
        elif recipe.photo in known and recipe.photo not in deleted:
            large = recipe.photo
        else:
            large = _photo_filename()
            extra = (
                RemotePhoto(
                    uid=large.removesuffix(".jpg"),
                    recipe_uid=recipe.uid,
                    filename=large,
                    name=str(len(recipe.gallery) + 1),
                    order_flag=len(recipe.gallery),
                    hash=hashlib.sha256(prepared.full).hexdigest().upper(),
                ),
                prepared.full,
            )

            if recipe.photo != server.photo:
                # A file the user dropped in, rather than a thumbnail of ours
                # with new bytes; it is tidied away once its copy is down.
                source = recipe.photo

        return _PhotoPlan(
            upload=replace(
                recipe,
                photo=_photo_filename(),
                photo_hash=hashlib.sha256(prepared.thumbnail).hexdigest().upper(),
                photo_large=large,
                photo_url=None,
            ),
            thumbnail=prepared.thumbnail,
            gallery=extra,
            source=source,
            note="its photo went to Paprika too"
            + (
                f", and {ATTACHMENTS_DIRNAME}/{source} gives way to the copies "
                "Paprika now holds"
                if source
                else ""
            ),
        )

    def _sync_photos(
        self,
        report: SyncReport,
        galleries: dict[str, list[RemotePhoto]],
        uids: Iterable[str],
    ) -> None:
        """Mirror the photos of the given recipes, as their base copies say."""
        for uid in sorted(uids):
            base = self._repository.read_base(uid)

            if base is not None:
                self._ensure_photos(report, base, galleries.get(uid, []))

    def _ensure_photos(
        self, report: SyncReport, recipe: RemoteRecipe, gallery: list[RemotePhoto]
    ) -> None:
        """Bring a recipe's thumbnail and gallery down into `attachments/`.

        Doing nothing is the common case: the state file says we already
        downloaded each photo in exactly this version, and the files are
        still on disk.  Everything else -- a recipe seen for the first time,
        a photo replaced in the app, an attachment tidied away by hand --
        ends with the photo downloaded again.  Gallery photos the server no
        longer has are removed.

        A failure is reported rather than raised.  One unreachable image
        should not stop a pull, and because the attachment is simply left
        missing, every later pull keeps trying until it succeeds.
        """
        repository = self._repository
        self._ensure_thumbnail(report, recipe)

        state = repository.read_photo_state(recipe.uid) or PhotoState()
        listed = {photo.filename for photo in gallery}

        for filename in sorted(set(state.gallery) - listed):
            repository.drop_gallery_photo(recipe.uid, filename)

        for photo in gallery:
            record = GalleryPhotoState(
                uid=photo.uid,
                name=photo.name or "",
                order_flag=photo.order_flag or 0,
                hash=photo.hash or "",
            )
            known = state.gallery.get(photo.filename)

            try:
                if (
                    known is not None
                    and known.hash == record.hash
                    and repository.attachment_path(photo.filename).is_file()
                ):
                    if replace(known, content_hash="") != record:
                        repository.record_gallery_photo(
                            recipe.uid, photo.filename, record
                        )
                    continue

                repository.write_gallery_photo(
                    recipe.uid,
                    photo.filename,
                    record,
                    self._download_gallery_photo(photo),
                )
            except (PaprikaError, PaprikaUserError) as e:
                report.record(
                    Action.SKIPPED,
                    recipe.uid,
                    recipe.name,
                    f"its gallery photo {photo.filename} could not be "
                    f"downloaded: {e}",
                )

        state = repository.read_photo_state(recipe.uid) or PhotoState()
        large = recipe.photo_large or ""

        repository.set_large_photo(
            recipe.uid, large if recipe.photo and large in state.gallery else ""
        )

    def _ensure_thumbnail(self, report: SyncReport, recipe: RemoteRecipe) -> None:
        """Bring a recipe's thumbnail down into `attachments/`, if it is not there.

        Called with what the recipe's *base copy* holds, since that is the
        photo its file's embed points at.
        """
        repository = self._repository

        try:
            if not recipe.photo:
                # The server's copy has no photo, so nothing should be left
                # sitting in `attachments/` claiming otherwise.
                repository.drop_thumbnail(recipe.uid)
                return

            state = repository.read_photo_state(recipe.uid)

            if (
                state is not None
                and state.photo == recipe.photo
                and state.photo_hash == recipe.photo_hash
                and repository.attachment_path(recipe.photo).is_file()
            ):
                return

            repository.write_photo(
                recipe.uid,
                recipe.photo,
                recipe.photo_hash,
                self._download_photo(recipe),
            )
        except (PaprikaError, PaprikaUserError) as e:
            report.record(
                Action.SKIPPED,
                recipe.uid,
                recipe.name,
                f"its photo could not be downloaded: {e}",
            )

    def _download_photo(self, recipe: RemoteRecipe) -> bytes:
        """Fetch the bytes behind a recipe's photo, renewing its link if stale.

        The `photo_url` a recipe carries is signed object storage that
        expires within hours, and the copy in hand has often been sitting
        longer than that -- in the response cache, or in a base copy healing
        an attachment days after it was pulled.  So the remembered link is
        only the first try; when it fails, or there never was one, the
        server is asked to sign a fresh one and that gets the last word.
        """
        if recipe.photo_url:
            try:
                return self._remote.download_photo(recipe.photo_url)
            except PaprikaError:
                pass

        url = self._remote.photo_download_url(recipe.uid)

        if not url:
            raise PaprikaError("the server did not say where to fetch it from")

        return self._remote.download_photo(url)

    def _download_gallery_photo(self, photo: RemotePhoto) -> bytes:
        """Fetch a gallery photo; its link has to be asked for each time."""
        url = self._remote.gallery_photo_url(photo.uid)

        if not url:
            raise PaprikaError("the server did not say where to fetch it from")

        return self._remote.download_photo(url)

    # -- Categories -------------------------------------------------------

    def _refresh_categories(self, dry_run: bool) -> CategoryNames:
        """Fetch the account's category names, and adopt them unless dry-running.

        A dry run still reads with whatever names the directory already
        has, since adopting new ones may mean rewriting files.
        """
        categories = self._remote.get_categories()

        if not dry_run:
            self._repository.update_categories(categories)

        return categories

    # -- Plumbing -----------------------------------------------------------

    def _fetch(self, uid: str, hash: str) -> RemoteRecipe:
        recipe = self._remote.get_recipe_by_id(uid, hash)
        self._notify(recipe.name)

        return recipe

    def _notify(self, name: str) -> None:
        if self._on_recipe is not None:
            self._on_recipe(name)


@dataclass(frozen=True)
class _PhotoPlan:
    """Everything one push needs to do about one recipe's own photo."""

    #: The recipe as it goes up: the file's fields, with the photo fields
    #: resolved to what the server should hold afterwards.
    upload: RemoteRecipe
    #: The thumbnail sent alongside the recipe, when a photo is going up.
    thumbnail: bytes | None = None
    #: A gallery photo to create for the full-size picture, when the photo
    #: going up is not already in the gallery.
    gallery: tuple[RemotePhoto, bytes] | None = None
    #: The user's own file the photo was read from, to tidy away afterwards.
    source: str = ""
    #: What to tell the user about all of the above.
    note: str = ""


@dataclass(frozen=True)
class _GalleryPlan:
    """Everything one push needs to do about one recipe's gallery."""

    #: Gallery photos to send: new ones with their image, changed ones with
    #: only their caption and position.
    uploads: tuple[tuple[RemotePhoto, bytes | None], ...] = ()
    #: Gallery photos to take down.
    deletions: tuple[RemotePhoto, ...] = ()
    #: The new server name of each file that goes up as a new photo.
    renames: Mapping[str, str] = field(default_factory=dict)
    #: The user's own files those were read from, to tidy away afterwards.
    sources: frozenset[str] = frozenset()
    #: What to tell the user about all of the above.
    note: str = ""


def _projection(photos: Iterable[RemotePhoto]) -> list[list[str]]:
    """A server gallery as the `[filename, caption]` list a file holds."""
    return normalize_gallery([[photo.filename, photo.name or ""] for photo in photos])


def _unknown_categories(
    entry: WorkingRecipe, categories: CategoryNames | None
) -> list[str]:
    """Categories a recipe names that the account does not have.

    Only categories the file adds are checked.  One the base copy already
    held may well be a category since deleted in the app; that is the
    server's own doing, and no reason to refuse an unrelated edit.
    """
    if categories is None or entry.recipe is None:
        return []

    held = set(entry.base.categories) if entry.base is not None else set()
    added = [value for value in entry.recipe.categories if value not in held]

    return categories.unknown(added)


def _categories_named(names: list[str]) -> str:
    quoted = [repr(name) for name in names]

    if len(quoted) == 1:
        return f"category {quoted[0]}"

    return f"categories {_and(quoted)}"


def _photo_filename() -> str:
    """A fresh server-side name for an image, in the style the app uses."""
    return f"{str(uuid.uuid4()).upper()}.jpg"


def _and(names: Iterable[str]) -> str:
    """Join names the way a person would say them aloud."""
    names = list(names)

    if len(names) < 2:
        return "".join(names)

    return f"{', '.join(names[:-1])} and {names[-1]}"
