"""Turning a photo the user dropped into a vault into what Paprika expects.

Paprika's own app never uploads a photo as-is.  It sends two derived images:
a square thumbnail that becomes the recipe's `photo` -- the image the app
shows in lists, and the one whose bytes `photo_hash` digests -- and a copy of
the picture itself, scaled down to fit a bounding box, that goes into the
recipe's photo gallery as `photo_large`.  We reproduce both here, at the same
sizes the app uses, so that a photo added from a directory is
indistinguishable from one added in the app.

The full-size image is passed through untouched when it already is what we
would have produced -- a JPEG within the size and byte caps, the right way
up, with no location in its metadata -- so that pushing a photo and pulling
it back yields the very bytes the user supplied whenever that is possible.

Anything Pillow can open is accepted, which with `pillow-heif` installed
includes the HEIC photos an iPhone takes.  Whatever comes in goes up as a
JPEG of at most `MAX_PHOTO_BYTES`: a photo that encodes larger at the
app's quality is re-encoded at lower quality, then scaled down, until it
fits.  Re-encoding drops the original's metadata, GPS position included.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

from PIL import Image, ImageOps, UnidentifiedImageError

from .exceptions import PaprikaUserError

try:  # HEIC/HEIF support, for photos straight off an iPhone.
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - an optional extra in principle
    pass
else:
    register_heif_opener()

#: The square thumbnail the app uploads as the recipe's `photo`, in pixels.
THUMBNAIL_SIZE: Final = 280

#: The bounding box the app scales the full image into before uploading it.
FULL_SIZE_LIMIT: Final = 2048

#: Matches the visible quality of the app's own uploads closely enough.
JPEG_QUALITY: Final = 85

#: The most a full-size photo may weigh when it goes up.
MAX_PHOTO_BYTES: Final = 1_000_000

#: How far quality is lowered before resolution is given up instead.
MIN_JPEG_QUALITY: Final = 70

#: The EXIF tags that say where a photo was taken.
GPS_IFD: Final = 0x8825


@dataclass(frozen=True)
class PreparedPhoto:
    """A user's photo, re-cut the way Paprika stores one."""

    #: A square JPEG for the recipe's `photo`; `photo_hash` digests these bytes.
    thumbnail: bytes
    #: The picture itself, as a JPEG no larger than `FULL_SIZE_LIMIT` a side.
    full: bytes


def prepare_photo(data: bytes) -> PreparedPhoto:
    """Cut a photo into the thumbnail and full-size JPEGs Paprika stores.

    Raises `PaprikaUserError` for bytes that are not a readable image.
    """
    try:
        with Image.open(io.BytesIO(data)) as source:
            # EXIF orientation 1 is "already the right way up"; anything else
            # means the pixels are stored rotated and the tag says how.
            exif = source.getexif()
            sideways = exif.get(0x0112, 1) != 1
            untouched = (
                source.format == "JPEG"
                and not sideways
                and max(source.size) <= FULL_SIZE_LIMIT
                and len(data) <= MAX_PHOTO_BYTES
                and GPS_IFD not in exif
            )

            image = ImageOps.exif_transpose(source) if sideways else source
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise PaprikaUserError(f"it could not be read as an image ({e})")

    return PreparedPhoto(
        thumbnail=_encode(ImageOps.fit(image, (THUMBNAIL_SIZE, THUMBNAIL_SIZE))),
        full=data if untouched else _encode_within_cap(image),
    )


def _encode_within_cap(image: Image.Image) -> bytes:
    """Encode at the app's size and quality, giving ground until it fits.

    Quality goes first, in small steps, since down to the floor the loss is
    hard to see; after that the picture is scaled down, which always ends.
    """
    limit = FULL_SIZE_LIMIT
    fitted = _fit_within(image, limit)

    for quality in range(JPEG_QUALITY, MIN_JPEG_QUALITY - 1, -5):
        encoded = _encode(fitted, quality)

        if len(encoded) <= MAX_PHOTO_BYTES:
            return encoded

    while len(encoded) > MAX_PHOTO_BYTES and limit > THUMBNAIL_SIZE:
        limit = int(limit * 0.85)
        encoded = _encode(_fit_within(image, limit), MIN_JPEG_QUALITY)

    return encoded


def _fit_within(image: Image.Image, limit: int) -> Image.Image:
    """Scale an image down (never up) to fit inside a `limit`-sized square."""
    if max(image.size) <= limit:
        return image

    scaled = image.copy()
    scaled.thumbnail((limit, limit))

    return scaled


def _encode(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    """Encode as JPEG, flattening any transparency onto white first."""
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality)

    return out.getvalue()
