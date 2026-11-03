"""Rendering recipes to markdown files, and reading them back again.

Paprika's own model is almost entirely unstructured -- of the fields on
`BaseRecipe`, only `categories` and `rating` hold anything but a string, and
the interesting ones (`ingredients`, `directions`, `notes`) are opaque text
blobs.  We therefore make no attempt to impose structure the source data does
not have: prose fields are written out verbatim and read back verbatim.

In particular we deliberately do *not* follow RecipeMD's convention of pulling
ingredient amounts out into an italicised prefix.  Real recipes contain lines
like `185 g wet ingredients: 2 large eggs, 3 large egg yolks, and enough water
to reach 185 g in total.` -- there is no amount to extract, and guessing at one
would make the round-trip lossy for no gain.

A recipe's photo is the one field with a visible home in the body that is not
prose: it is written as an ordinary markdown image embed directly beneath the
title, pointing into the `attachments/` folder beside the recipe files.  The
embed is the photo's representation in the document, and it round-trips like
any other field -- which is what will eventually let deleting the line mean
"remove the photo" and writing one mean "add this photo".  The photo's
server-side bookkeeping (`photo_hash` and friends) has no place in a file a
human reads, and lives only in the base copy; see `HIDDEN_FIELDS`.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, TypeVar
from urllib.parse import quote, unquote

from yaml import YAMLError

from .categories import CategoryNames
from .exceptions import PaprikaUserError
from .utils import dump_yaml, load_yaml

if TYPE_CHECKING:
    from .recipe import BaseRecipe

T = TypeVar("T", bound="BaseRecipe")

FRONTMATTER_DELIMITER: Final = "---"

#: The document's title, and the level at which sections are delimited.
TITLE_PREFIX: Final = "# "
HEADING_PREFIX: Final = "## "

#: Written as the document's `# ` title rather than as a frontmatter field.
TITLE_FIELD: Final = "name"

#: The frontmatter field that says which recipe a file is.
UID_FIELD: Final = "uid"

#: Suffixes that make a frontmatter key look like somebody's idea of a uid.
#: Deliberately not a bare `uid` suffix, which would also match a vault's own
#: `uuid:`.
UID_SUFFIXES: Final = ("_uid", "-uid")

#: Written directly beneath the title, with no heading of its own.
DESCRIPTION_FIELD: Final = "description"

#: Holds category uids in the recipe, and category names in the file; see
#: `categories`.
CATEGORIES_FIELD: Final = "categories"

#: Written as an image embed between the title and the description, holding
#: the filename of the recipe's photo inside `attachments/`.
PHOTO_FIELD: Final = "photo"

#: Where photos live, as a sibling of the recipe files.  The embed points
#: here, and it is a folder a note vault will already feel at home with.
ATTACHMENTS_DIRNAME: Final = "attachments"

#: Photo bookkeeping that travels with the recipe but has no place in the
#: file: server-side hashes and URLs that mean nothing to a reader and would
#: only rot in their frontmatter.  They live in the base copy instead, so
#: they are never lost -- just never shown.
HIDDEN_FIELDS: Final = frozenset({"photo_hash", "photo_large", "photo_url"})

#: Everything about a recipe's photo, the embed and the bookkeeping alike.
PHOTO_FIELDS: Final = frozenset({PHOTO_FIELD}) | HIDDEN_FIELDS

#: A line that is nothing but an image embed pointing into `attachments/`.
#: This is the one line shape besides a `## ` heading that belongs to the
#: format rather than to the recipe's own text; see `_escape_prose`.  The
#: target runs greedily to the line's final `)` so that a filename someone
#: wrote by hand -- parentheses, spaces and all -- still reads as the photo
#: it plainly is; our own renderer percent-encodes all of that anyway.
PHOTO_EMBED: Final = re.compile(rf"^!\[[^\]]*\]\({ATTACHMENTS_DIRNAME}/(.+)\)\s*$")

#: The same embed wrapped in a link to the full-size picture, which is how
#: the photo is written when we have one; see `DocumentFormat.photo_links`.
#: The image inside is the photo, and the link target is presentation only.
PHOTO_LINKED: Final = re.compile(
    rf"^\[!\[[^\]]*\]\({ATTACHMENTS_DIRNAME}/(.+?)\)\]"
    rf"\({ATTACHMENTS_DIRNAME}/(.+)\)\s*$"
)

#: Written as `## ` sections, in this order.  Every recipe field not named
#: here or above is written to the frontmatter instead.
SECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("Ingredients", "ingredients"),
    ("Directions", "directions"),
    ("Notes", "notes"),
    ("Nutritional Information", "nutritional_info"),
)

#: The recipe's photo gallery: a list of `[filename, caption]` pairs, in
#: order, written as a `## Photos` section after every other section: a
#: bullet list of links to the pictures, captioned, so that a previewer shows
#: a tidy list rather than a wall of full-size images.  It is not part of Paprika's recipe JSON -- the gallery is a
#: separate collection on the server -- but it travels with the recipe so that
#: change detection, merging and restoring treat it like any other field.
GALLERY_FIELD: Final = "gallery"
GALLERY_HEADING: Final = "Photos"

#: One line of the `## Photos` section: a link (or, for an uncaptioned
#: photo, an image) pointing into `attachments/`, optionally as a bullet.
#: The caption may hold backslash-escaped brackets; see `_gallery_line`.
GALLERY_LINE: Final = re.compile(
    r"^(?:[-*+] )?!?\[((?:\\.|[^\]\\])*)\]\(" + ATTACHMENTS_DIRNAME + r"/(.+)\)\s*$"
)

#: Fields rendered as a markdown bullet list, one bullet per line of the blob.
LIST_FIELDS: Final = frozenset({"ingredients"})

#: Bullet characters we will accept when reading a list field back in.  We
#: only ever *write* `-`, but a human editing the file might reasonably type
#: any of these.
BULLETS: Final = ("- ", "* ", "+ ")

BODY_FIELDS: Final = frozenset(
    {TITLE_FIELD, DESCRIPTION_FIELD, PHOTO_FIELD, GALLERY_FIELD}
    | {field for _, field in SECTIONS}
)

_HEADINGS_BY_TITLE: Final = dict(SECTIONS)


@dataclass(frozen=True)
class ExtraSection:
    """A `## ` section of a recipe file that is not one of ours."""

    #: The recipe field whose section this one followed, so that it can be put
    #: back where its author left it.  Empty if it came before any of ours.
    anchor: str
    #: The section verbatim, its heading included.
    text: str


@dataclass(frozen=True)
class Extras:
    """The parts of a recipe file that belong to its reader, not to Paprika.

    A recipe file is meant to be a good citizen of whatever directory it lands
    in, and somewhere like an Obsidian vault that means the file is not
    exclusively ours: a user may add `tags:` to the frontmatter and a
    `## Substitutions` section of their own to the body.  Paprika has nowhere
    to put either, so we never send them anywhere -- but we do carry them
    through unchanged whenever we rewrite the file.
    """

    frontmatter: dict[str, Any] = field(default_factory=dict)
    sections: tuple[ExtraSection, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.frontmatter or self.sections)

    def describe(self) -> str:
        """Name what is here, briefly enough to sit at the end of a line.

        Whoever wrote this into their file should be told plainly that it is
        staying put rather than going to Paprika -- silently keeping it is
        only marginally better than silently dropping it, since either way
        they are left guessing what happened to it.
        """
        headings = [section.text.split("\n", 1)[0] for section in self.sections]
        keys = [f"{key}:" for key in sorted(self.frontmatter)]

        return ", ".join([*headings, *keys])


@dataclass(frozen=True)
class DocumentFormat:
    """How a recipe is spelled in a file.

    A recipe file is meant to be a good citizen of whatever directory it lands
    in, and in a note vault that means our frontmatter is sharing a namespace
    with the vault's own conventions.  `rating:`, `source:`, `categories:` and
    `created:` are all things a vault may already mean something else by, so a
    directory can be cloned with a prefix -- `paprika_rating:`, and so on.

    The prefix is strict in both directions.  With one configured, an
    *unprefixed* frontmatter field belongs to the user: it is carried through
    and never uploaded.  Accepting unprefixed fields as a fallback would
    defeat the whole point, because the reason to set a prefix is that the
    vault already has a `rating:` of its own -- and reading that as the
    recipe's rating would upload it.

    A prefixed field we do not recognise is kept as the user's too, rather
    than discarded: it costs nothing and it means a file written by a later
    version of this program loses nothing by being read by an earlier one.

    `categories` is the account's list of category names, which lets the
    file spell a recipe's categories by name while the recipe itself holds
    their uids.  With no list -- an archive, which already holds names, or a
    directory that has never pulled one -- values pass through as they are.
    """

    frontmatter_prefix: str = ""
    categories: CategoryNames = field(default_factory=CategoryNames)
    #: Where the photo embed links to: each recipe's thumbnail mapped to its
    #: full-size gallery copy.  The embed shows the thumbnail, which previews
    #: well anywhere, and clicking it opens the picture itself.  The link is
    #: presentation only: reading a file back ignores where it points.  See
    #: `Repository.photo_links`.
    photo_links: Mapping[str, str] = field(default_factory=dict)

    def render(self, recipe: BaseRecipe, extras: Extras | None = None) -> str:
        """Render a recipe as a markdown document with YAML frontmatter."""
        extras = extras if extras is not None else Extras()
        data = recipe.as_dict()

        if isinstance(data.get(CATEGORIES_FIELD), list):
            data[CATEGORIES_FIELD] = self.categories.display_all(data[CATEGORIES_FIELD])

        frontmatter: dict[str, Any] = dict(extras.frontmatter)
        frontmatter.update(
            {
                f"{self.frontmatter_prefix}{key}": value
                for key, value in data.items()
                if key not in BODY_FIELDS and key not in HIDDEN_FIELDS
            }
        )

        return _render(data, frontmatter, extras, self.photo_links)

    def parse(self, content: str, recipe_class: type[T]) -> tuple[T, Extras]:
        """Read a document as both a recipe and whatever else its file holds."""
        frontmatter, body = _split_frontmatter(content)
        fields, sections = _parse_body(body)
        known = {field.name for field in recipe_class.get_all_fields()}

        ours: dict[str, Any] = {}
        theirs: dict[str, Any] = {}

        for key, value in frontmatter.items():
            name = self._field_for(key, known)

            if name is None:
                theirs[key] = value
            else:
                ours[name] = value

        data: dict[str, Any] = dict(ours)
        data.update(fields)

        if isinstance(data.get(CATEGORIES_FIELD), list):
            data[CATEGORIES_FIELD] = self.categories.resolve_all(data[CATEGORIES_FIELD])

        # A file that does not name a uid does not have one.  Without this the
        # dataclass's default would invent a *different* uid every time the
        # same file was read, which makes "this recipe has no identity yet"
        # indistinguishable from "this recipe has one" -- and silently so,
        # since nothing downstream can tell an invented uid from a real one.
        data.setdefault(UID_FIELD, "")

        return recipe_class.from_dict(data), Extras(
            frontmatter=theirs, sections=sections
        )

    def parse_recipe(self, content: str, recipe_class: type[T]) -> T:
        """Read the recipe out of a document produced by `render`."""
        return self.parse(content, recipe_class)[0]

    def read_extras(self, content: str, recipe_class: type[BaseRecipe]) -> Extras:
        """Read the parts of a document that are not the recipe."""
        return self.parse(content, recipe_class)[1]

    def find_lossy_fields(
        self, recipe: BaseRecipe, extras: Extras | None = None
    ) -> list[str]:
        """Return the parts of a recipe that do not survive a round-trip.

        Under normal circumstances this is empty; it is non-empty when a recipe
        contains something our format cannot express unambiguously.  Callers
        can use it to refuse to write a file they would not be able to read
        back correctly.
        """
        rendered = self.render(recipe, extras)
        reparsed = self.parse_recipe(rendered, type(recipe))

        before = recipe.as_dict()
        after = reparsed.as_dict()

        lossy = sorted(
            key
            for key in before
            # The photo's bookkeeping deliberately never enters the file, so
            # its absence on the way back out is not a loss; the base copy is
            # what carries it.  See `HIDDEN_FIELDS`.
            if key not in HIDDEN_FIELDS and before[key] != after.get(key)
        )

        if extras is not None and self.read_extras(rendered, type(recipe)) != extras:
            lossy.append("the sections you added yourself")

        return lossy

    def _field_for(self, key: str, known: set[str]) -> str | None:
        """Which recipe field a frontmatter key holds, if it holds one at all."""
        if not key.startswith(self.frontmatter_prefix):
            return None

        name = key[len(self.frontmatter_prefix) :]

        return name if name in known else None


#: What a directory cloned without a prefix uses, and what anything with no
#: directory to ask -- an archive, say -- has to assume.
DEFAULT_FORMAT: Final = DocumentFormat()


def render_recipe(recipe: BaseRecipe, extras: Extras | None = None) -> str:
    """Render a recipe as a markdown document, in the unprefixed format."""
    return DEFAULT_FORMAT.render(recipe, extras)


def parse_document(content: str, recipe_class: type[T]) -> tuple[T, Extras]:
    """Read a document in the unprefixed format."""
    return DEFAULT_FORMAT.parse(content, recipe_class)


def parse_recipe(content: str, recipe_class: type[T]) -> T:
    """Read the recipe out of a document in the unprefixed format."""
    return DEFAULT_FORMAT.parse_recipe(content, recipe_class)


def read_extras(content: str, recipe_class: type[BaseRecipe]) -> Extras:
    """Read the parts of an unprefixed document that are not the recipe."""
    return DEFAULT_FORMAT.read_extras(content, recipe_class)


def find_lossy_fields(recipe: BaseRecipe, extras: Extras | None = None) -> list[str]:
    """Round-trip an unprefixed document; see `DocumentFormat`."""
    return DEFAULT_FORMAT.find_lossy_fields(recipe, extras)


def _render(
    data: dict[str, Any],
    frontmatter: dict[str, Any],
    extras: Extras,
    photo_links: Mapping[str, str] | None = None,
) -> str:
    out = io.StringIO()
    out.write(f"{FRONTMATTER_DELIMITER}\n")
    dump_yaml(frontmatter, out)
    out.write(f"{FRONTMATTER_DELIMITER}\n\n")
    out.write(f"# {data.get(TITLE_FIELD, '')}\n")

    photo = data.get(PHOTO_FIELD) or ""
    if photo:
        embed = _photo_embed(photo, data.get(TITLE_FIELD) or "")
        full = (photo_links or {}).get(photo)

        if full:
            embed = f"[{embed}]({ATTACHMENTS_DIRNAME}/{quote(full, safe='')})"

        out.write(f"\n{embed}\n")

    description = _render_field(DESCRIPTION_FIELD, data.get(DESCRIPTION_FIELD) or "")
    if description:
        out.write(f"\n{description}\n")

    placed = {""}
    _render_extras(out, extras, "")

    for title, field_name in SECTIONS:
        value = data.get(field_name) or ""
        if not value:
            continue

        out.write(f"\n## {title}\n\n")
        out.write(_render_field(field_name, value))
        out.write("\n")

        placed.add(field_name)
        _render_extras(out, extras, field_name)

    gallery = data.get(GALLERY_FIELD) or []
    if gallery:
        out.write(f"\n## {GALLERY_HEADING}\n\n")
        out.write("\n".join(_gallery_line(*entry) for entry in gallery))
        out.write("\n")

        placed.add(GALLERY_FIELD)
        _render_extras(out, extras, GALLERY_FIELD)

    # A section anchored to one of ours that this recipe no longer has still
    # needs to go somewhere; the end is the only honest place left for it.
    for section in extras.sections:
        if section.anchor not in placed:
            out.write(f"\n{section.text}\n")

    return out.getvalue()


def foreign_uid_key(extras: Extras) -> str:
    """A frontmatter key that looks like a uid we did not recognise as ours.

    A recipe file with no uid is ordinarily just a recipe nobody has synced
    yet.  But a file that carries something *called* a uid while reading as
    untracked is a different thing entirely: it has an identity we cannot see.
    That is what a mis-set `frontmatter_prefix` looks like from in here, and
    treating it as a new recipe would upload a duplicate of something we
    already have -- while its base copy, now unclaimed, looked like a deletion.
    """
    for key in sorted(extras.frontmatter):
        lowered = key.lower()

        if lowered == UID_FIELD or lowered.endswith(UID_SUFFIXES):
            return key

    return ""


def documents_differ(content: str, rendered: str) -> bool:
    """Does a recipe file say something different from a rendering of a recipe?

    The two halves of the document are compared differently, on purpose.

    The body is compared as *text*.  Our markdown encoding of a recipe's prose
    is a bespoke transformation, and comparing the text is what keeps change
    detection independent of how faithfully that transformation round-trips:
    both sides pass through it identically, so any normalisation cancels out.

    The frontmatter is compared as *data*.  It is ordinary YAML, which we did
    not invent and which `safe_load` reads back faithfully -- and there are
    many ways to spell the same mapping.  A user who types `tags: [dinner]`
    into their editor rather than the block form we would have written has not
    edited the recipe, and should not be told they have.
    """
    try:
        theirs, their_body = _split_frontmatter(content)
    except PaprikaUserError:
        # Not a document we could have written; treat it as changed and let
        # whoever tries to read it produce the useful error message.
        return True

    ours, our_body = _split_frontmatter(rendered)

    return ours != theirs or our_body != their_body


def normalize_recipe(recipe: T) -> T:
    """Return a copy of `recipe` in the form our markdown can represent exactly.

    Markdown has nowhere to put blank lines at either end of a section -- the
    whitespace on each side of a heading is ours, not the recipe's -- so a
    field beginning or ending in a newline would come back shorter than it
    went out.  Rather than treat that as a failure, we flatten it on the way
    in, so that the base copy and the working file always agree about what
    the recipe says.

    Line endings get the same treatment.  Paprika stores whatever the client
    that saved the recipe used -- `\r\n` from Windows, bare `\r` from older
    apps -- and a working file is read back in universal-newlines mode, which
    folds all of those to `\n`.  Writing a `\r` we would never read back is
    exactly the mismatch this function exists to prevent, so every line
    ending is folded to `\n` before anything is written.

    Absent text gets one more courtesy: Paprika spells an empty field as
    `null`, and our markdown can only spell it as an absent section or embed,
    which reads back as `""`.  The two mean the same thing, so `null` is
    flattened on the way in rather than reported as a loss.

    That is all the normalisation we perform. It discards nothing a cook would
    notice, and it only ever reaches the server for a field the user edited
    anyway.
    """
    changes: dict[str, Any] = {}

    if hasattr(recipe, GALLERY_FIELD):
        gallery = normalize_gallery(getattr(recipe, GALLERY_FIELD))

        if gallery != getattr(recipe, GALLERY_FIELD):
            changes[GALLERY_FIELD] = gallery

    for field_name in BODY_FIELDS - {GALLERY_FIELD}:
        value = getattr(recipe, field_name, None)

        if value is None:
            changes[field_name] = ""
            continue

        if not isinstance(value, str):
            continue

        normalized = _normalize_newlines(value)
        normalized = (
            normalized.strip() if field_name == TITLE_FIELD else normalized.strip("\n")
        )

        if normalized != value:
            changes[field_name] = normalized

    return replace(recipe, **changes) if changes else recipe


def normalize_gallery(gallery: Any) -> list[list[str]]:
    """A gallery as `[filename, caption]` lists, in the form a file can hold.

    A caption is a single line of alt text, so line breaks inside one are
    folded to spaces and its ends are trimmed; Paprika stores whatever its
    app was given.
    """
    result = []

    for filename, caption in gallery or []:
        caption = " ".join(_normalize_newlines(caption or "").split("\n")).strip()
        result.append([str(filename), caption])

    return result


def _normalize_newlines(value: str) -> str:
    """Fold every flavour of line ending to `\n`, as reading a file would."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _render_field(field_name: str, value: str) -> str:
    if field_name not in LIST_FIELDS:
        return _escape_prose(value.rstrip("\n"))

    # Every line of a list field is prefixed with a bullet, which already
    # stops it being mistaken for one of our headings.
    return "\n".join(f"- {line}" if line else "" for line in value.split("\n"))


def _parse_field(field_name: str, value: str) -> str:
    if field_name not in LIST_FIELDS:
        return _unescape_prose(value)

    lines = []
    for line in value.split("\n"):
        for bullet in BULLETS:
            if line.startswith(bullet):
                line = line[len(bullet) :]
                break

        lines.append(line)

    return "\n".join(lines)


def _is_heading(line: str) -> bool:
    return line.startswith(HEADING_PREFIX)


def _photo_embed(photo: str, name: str) -> str:
    """The image embed line for a recipe's photo.

    The filename is percent-encoded, which both keeps the link valid markdown
    whatever the name holds and makes the encoding reversible on the way back
    in.  The alt text is only a courtesy to screen readers and broken links;
    it is regenerated on every render and never parsed.
    """
    alt = "".join(
        character
        for character in (f"Photo of {name}" if name else "Photo")
        if character not in "[]\\\n"
    )

    return f"![{alt}]({ATTACHMENTS_DIRNAME}/{quote(photo, safe='')})"


def _gallery_line(filename: str, caption: str) -> str:
    """One line of the `## Photos` section: a bullet linking to the picture.

    The caption is the link text.  An uncaptioned photo is written as an
    image instead, since an empty link would be invisible in a preview.
    Backslashes and brackets in the caption are escaped, so that any caption
    reads back exactly; the filename is percent-encoded, as for the photo.
    """
    text = re.sub(r"([\\\[\]])", r"\\\1", caption)
    target = f"({ATTACHMENTS_DIRNAME}/{quote(filename, safe='')})"

    return f"- [{text}]{target}" if text else f"- ![]{target}"


def _parse_gallery(text: str) -> list[list[str]]:
    """Read the `## Photos` section back into `[filename, caption]` pairs."""
    gallery = []

    for line in text.split("\n"):
        if not line.strip():
            continue

        match = GALLERY_LINE.match(line)

        if match is None:
            raise PaprikaUserError(
                f"`## {GALLERY_HEADING}` may only hold links to (or images "
                f"of) files in {ATTACHMENTS_DIRNAME}/, one per line, but "
                f"found {line!r}."
            )

        caption = re.sub(r"\\(.)", r"\1", match.group(1))
        gallery.append([unquote(match.group(2)), caption])

    return gallery


def _split_photo(text: str) -> tuple[str, str]:
    """Take the photo embed off the top of the description block.

    Only the first line of the block can be a photo -- that is where `_render`
    puts it, directly beneath the title.  An image the user pastes anywhere
    else in their prose is their text, not our markup, and stays put.
    """
    first, _, rest = text.partition("\n")
    photo = _photo_line(first)

    if photo is None:
        return "", text

    return unquote(photo), rest.strip("\n")


def _photo_line(line: str) -> str | None:
    """The photo a line embeds, linked or not; None if it is not an embed."""
    match = PHOTO_LINKED.match(line) or PHOTO_EMBED.match(line)

    return match.group(1) if match is not None else None


def _escape_prose(value: str) -> str:
    """Stop prose from being mistaken for markup that belongs to the format.

    Every `## ` line we write out of a recipe's prose is escaped, not just the
    ones that collide with our own section names.  That is what makes the
    format unambiguous in the other direction: an *unescaped* `## ` in a file
    is always a section boundary, so a section the user added themselves can
    be told apart from a line of directions that merely looks like one.

    A line that looks like one of our photo embeds gets the same treatment,
    for the same reason: an unescaped embed at the top of the document must
    always mean the recipe's photo, never a coincidence of its description.

    Backslash-escaping the first character is standard CommonMark and renders
    as the literal text, so the file still reads correctly to a human and to
    any previewer.
    """
    return "\n".join(
        f"\\{line}" if _is_heading(line) or _photo_line(line) is not None else line
        for line in value.split("\n")
    )


def _unescape_prose(value: str) -> str:
    return "\n".join(
        line[1:] if _is_escaped(line) else line for line in value.split("\n")
    )


def _is_escaped(line: str) -> bool:
    if not line.startswith("\\"):
        return False

    return _is_heading(line[1:]) or _photo_line(line[1:]) is not None


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    lines = content.split("\n")

    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        raise PaprikaUserError(
            "Recipe does not begin with a YAML frontmatter block; expected the "
            f"first line to be {FRONTMATTER_DELIMITER!r}."
        )

    remainder = lines[1:]

    for index, line in enumerate(remainder):
        if line.strip() != FRONTMATTER_DELIMITER:
            continue

        try:
            frontmatter = load_yaml("\n".join(remainder[:index])) or {}
        except YAMLError as e:
            raise PaprikaUserError(f"Recipe's YAML frontmatter could not be read: {e}")

        if not isinstance(frontmatter, dict):
            raise PaprikaUserError(
                "Recipe's YAML frontmatter should be a mapping of field names "
                f"to values, but found {type(frontmatter).__name__}."
            )

        return frontmatter, "\n".join(remainder[index + 1 :])

    raise PaprikaUserError(
        "Recipe's YAML frontmatter block was opened but never closed; "
        f"expected a line reading {FRONTMATTER_DELIMITER!r}."
    )


def _parse_body(body: str) -> tuple[dict[str, Any], tuple[ExtraSection, ...]]:
    """Split a document body into the recipe's fields and everything else."""
    result: dict[str, Any] = {}
    seen_title = False

    # Each block is a heading and the lines beneath it; the first block, with
    # no heading, is the description.
    blocks: list[tuple[str, list[str]]] = [("", [])]

    for line in body.split("\n"):
        if not seen_title and line.startswith(TITLE_PREFIX):
            result[TITLE_FIELD] = line[len(TITLE_PREFIX) :].strip()
            seen_title = True
            blocks = [("", [])]
            continue

        if _is_heading(line):
            blocks.append((line[len(HEADING_PREFIX) :].strip(), []))
            continue

        blocks[-1][1].append(line)

    extras: list[ExtraSection] = []
    anchor = ""

    for heading, lines in blocks:
        text = "\n".join(lines).strip("\n")

        if not heading:
            photo, text = _split_photo(text)

            if photo:
                result[PHOTO_FIELD] = photo
            if text:
                result[DESCRIPTION_FIELD] = _parse_field(DESCRIPTION_FIELD, text)
            continue

        if heading == GALLERY_HEADING:
            result[GALLERY_FIELD] = _parse_gallery(text)
            anchor = GALLERY_FIELD
            continue

        field_name = _HEADINGS_BY_TITLE.get(heading)

        if field_name is None:
            heading_line = f"{HEADING_PREFIX}{heading}"
            extras.append(
                ExtraSection(
                    anchor, f"{heading_line}\n\n{text}" if text else heading_line
                )
            )
            continue

        result[field_name] = _parse_field(field_name, text)
        anchor = field_name

    return result, tuple(extras)


def _render_extras(out: io.StringIO, extras: Extras, anchor: str) -> None:
    for section in extras.sections:
        if section.anchor == anchor:
            out.write(f"\n{section.text}\n")
