import io

import pytest
from PIL import Image
from test_sync import FakeAccount, actions, image_bytes, make_recipe

from paprika_recipes.exceptions import PaprikaUserError
from paprika_recipes.images import MAX_PHOTO_BYTES, prepare_photo
from paprika_recipes.markdown import DocumentFormat, parse_recipe, render_recipe
from paprika_recipes.remote import RemotePhoto, RemoteRecipe
from paprika_recipes.repository import Repository, Status
from paprika_recipes.sync import Action, Syncer


@pytest.fixture
def repository(tmp_path) -> Repository:
    return Repository.initialize(tmp_path)


def add_gallery_photo(
    account: FakeAccount, recipe: str, name: str, caption: str, order: int, data: bytes
) -> str:
    """Put a photo into a recipe's gallery the way the app would."""
    filename = f"{name}.jpg"
    url = f"https://photos.example/{filename}"
    account.photos[url] = data
    account.gallery[name] = RemotePhoto(
        uid=name,
        recipe_uid=recipe,
        filename=filename,
        name=caption,
        order_flag=order,
        hash=f"hash-{name}",
        photo_url=url,
    )

    return filename


@pytest.fixture
def account() -> FakeAccount:
    account = FakeAccount(make_recipe("A"), make_recipe("B"))
    account.give_photo("A", b"thumbnail")
    add_gallery_photo(account, "A", "card", "Original recipe card", 1, b"card bytes")

    return account


def content(repository: Repository, uid: str = "A") -> str:
    return repository.paths_by_uid()[uid].read_text(encoding="utf-8")


def edit(repository: Repository, old: str, new: str, uid: str = "A") -> None:
    path = repository.paths_by_uid()[uid]
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def live_gallery(account: FakeAccount, uid: str = "A") -> list[RemotePhoto]:
    return sorted(
        (p for p in account.gallery.values() if p.recipe_uid == uid and not p.deleted),
        key=lambda p: p.order_flag,
    )


class TestTheFormat:
    def test_the_photo_links_to_its_full_size_copy(self):
        rendered = DocumentFormat(photo_links={"T.jpg": "L.jpg"}).render(
            RemoteRecipe(name="Stew", photo="T.jpg")
        )

        assert "[![Photo of Stew](attachments/T.jpg)](attachments/L.jpg)" in rendered

    def test_the_link_is_ignored_when_reading(self):
        recipe = parse_recipe(
            "---\n---\n\n# Stew\n\n[![x](attachments/T.jpg)](attachments/X.jpg)\n",
            RemoteRecipe,
        )

        assert recipe.photo == "T.jpg"

    def test_the_gallery_is_a_list_of_captioned_links(self):
        rendered = render_recipe(
            RemoteRecipe(name="Stew", gallery=[["a.jpg", "Front"], ["b.jpg", ""]])
        )

        assert (
            "## Photos\n\n- [Front](attachments/a.jpg)\n- ![](attachments/b.jpg)\n"
            in (rendered)
        )

    @pytest.mark.parametrize(
        "caption", ["Plain", "With [brackets]", "Back\\slash", "", "Ünïcode"]
    )
    def test_captions_round_trip(self, caption):
        recipe = RemoteRecipe(name="Stew", gallery=[["a b.jpg", caption]])

        assert parse_recipe(render_recipe(recipe), RemoteRecipe).gallery == [
            ["a b.jpg", caption]
        ]

    def test_images_and_bare_lines_are_read_too(self):
        recipe = parse_recipe(
            "---\n---\n\n# Stew\n\n## Photos\n\n![Front](attachments/a.jpg)\n"
            "[Back](attachments/b.jpg)\n",
            RemoteRecipe,
        )

        assert recipe.gallery == [["a.jpg", "Front"], ["b.jpg", "Back"]]

    def test_anything_else_in_the_section_is_refused(self):
        with pytest.raises(PaprikaUserError, match="## Photos"):
            parse_recipe(
                "---\n---\n\n# Stew\n\n## Photos\n\nsome prose\n", RemoteRecipe
            )


class TestPulling:
    def test_mirrors_the_thumbnail_and_every_gallery_photo(self, repository, account):
        Syncer(repository, account).pull()

        attachments = repository.attachments_dir
        assert (attachments / "A-1.jpg").read_bytes() == b"thumbnail"
        assert (attachments / "A-large-1.jpg").read_bytes() == b"thumbnail"
        assert (attachments / "card.jpg").read_bytes() == b"card bytes"

    def test_the_file_links_the_thumbnail_and_lists_the_gallery(
        self, repository, account
    ):
        Syncer(repository, account).pull()

        text = content(repository)
        assert "(attachments/A-1.jpg)](attachments/A-large-1.jpg)" in text
        assert "- [1](attachments/A-large-1.jpg)\n" in text
        assert "- [Original recipe card](attachments/card.jpg)\n" in text

    def test_leaves_the_recipe_clean(self, repository, account):
        Syncer(repository, account).pull()

        assert {entry.status for entry in repository.status()} == {Status.UNCHANGED}
        assert not Syncer(repository, account).pull()

    def test_a_photo_added_in_the_app_arrives(self, repository, account):
        Syncer(repository, account).pull()
        add_gallery_photo(account, "A", "side", "Side view", 2, b"side bytes")

        report = Syncer(repository, account).pull()

        assert actions(report) == {"Recipe A": Action.UPDATED}
        assert (repository.attachments_dir / "side.jpg").read_bytes() == b"side bytes"
        assert "- [Side view](attachments/side.jpg)" in content(repository)

    def test_a_photo_removed_in_the_app_goes(self, repository, account):
        Syncer(repository, account).pull()
        account.gallery["card"] = RemotePhoto(
            **{**account.gallery["card"].__dict__, "deleted": True}
        )

        Syncer(repository, account).pull()

        assert not (repository.attachments_dir / "card.jpg").exists()
        assert "card.jpg" not in content(repository)

    def test_a_tidied_away_gallery_file_is_healed(self, repository, account):
        Syncer(repository, account).pull()
        (repository.attachments_dir / "card.jpg").unlink()

        Syncer(repository, account).pull()

        assert (repository.attachments_dir / "card.jpg").read_bytes() == b"card bytes"


class TestPushing:
    def test_a_new_line_uploads_a_new_gallery_photo(self, repository, account):
        Syncer(repository, account).pull()
        (repository.attachments_dir / "plated.jpg").write_bytes(image_bytes("red"))
        edit(
            repository,
            "- [Original recipe card](attachments/card.jpg)\n",
            "- [Original recipe card](attachments/card.jpg)\n"
            "- [Plated](attachments/plated.jpg)\n",
        )

        report = Syncer(repository, account).push()

        assert actions(report) == {"Recipe A": Action.UPLOADED}
        new = live_gallery(account)[-1]
        assert (new.name, new.order_flag) == ("Plated", 2)
        assert not (repository.attachments_dir / "plated.jpg").exists()
        assert (repository.attachments_dir / new.filename).is_file()
        assert f"- [Plated](attachments/{new.filename})" in content(repository)
        assert {entry.status for entry in repository.status()} == {Status.UNCHANGED}

    def test_a_gallery_change_alone_leaves_the_recipe_alone(self, repository, account):
        Syncer(repository, account).pull()
        before = account.recipes["A"]["hash"]
        edit(repository, "[Original recipe card]", "[The card]")

        Syncer(repository, account).push()

        assert account.recipes["A"]["hash"] == before
        assert account.gallery["card"].name == "The card"

    def test_a_removed_line_takes_the_photo_down(self, repository, account):
        Syncer(repository, account).pull()
        edit(repository, "- [Original recipe card](attachments/card.jpg)\n", "")

        Syncer(repository, account).push()

        assert account.gallery["card"].deleted
        assert not (repository.attachments_dir / "card.jpg").exists()

    def test_reordering_moves_the_photos(self, repository, account):
        Syncer(repository, account).pull()
        edit(
            repository,
            "- [1](attachments/A-large-1.jpg)\n"
            "- [Original recipe card](attachments/card.jpg)\n",
            "- [Original recipe card](attachments/card.jpg)\n"
            "- [1](attachments/A-large-1.jpg)\n",
        )

        Syncer(repository, account).push()

        assert [p.filename for p in live_gallery(account)] == [
            "card.jpg",
            "A-large-1.jpg",
        ]

    def test_replaced_bytes_go_up_as_a_new_photo(self, repository, account):
        Syncer(repository, account).pull()
        data = image_bytes("green")
        (repository.attachments_dir / "card.jpg").write_bytes(data)

        assert [entry.changed_fields() for entry in repository.status()] == [
            ["gallery"],
            [],
        ]

        Syncer(repository, account).push()

        assert account.gallery["card"].deleted
        new = live_gallery(account)[-1]
        assert new.name == "Original recipe card"
        assert (repository.attachments_dir / new.filename).read_bytes() == data

    def test_a_server_side_gallery_change_blocks_the_push(self, repository, account):
        Syncer(repository, account).pull()
        edit(repository, "[Original recipe card]", "[The card]")
        add_gallery_photo(account, "A", "side", "Side view", 2, b"side bytes")

        report = Syncer(repository, account).push()

        (conflict,) = report.conflicts
        assert "pull first" in conflict.detail

    def test_a_missing_file_stops_only_that_recipe(self, repository, account):
        Syncer(repository, account).pull()
        edit(
            repository,
            "- [Original recipe card](attachments/card.jpg)\n",
            "- [Original recipe card](attachments/card.jpg)\n"
            "- [Ghost](attachments/ghost.jpg)\n",
        )

        report = Syncer(repository, account).push()

        (skipped,) = report.of(Action.SKIPPED)
        assert "ghost.jpg" in skipped.detail
        assert len(live_gallery(account)) == 2

    def test_a_photo_picked_from_the_gallery_is_not_uploaded_twice(
        self, repository, account
    ):
        account.photos["https://photos.example/card.jpg"] = image_bytes("blue")
        Syncer(repository, account).pull()
        edit(
            repository,
            "[![Photo of Recipe A](attachments/A-1.jpg)](attachments/A-large-1.jpg)",
            "![Photo of Recipe A](attachments/card.jpg)",
        )

        Syncer(repository, account).push()

        assert account.recipes["A"]["photo_large"] == "card.jpg"
        assert len(live_gallery(account)) == 2


def jpeg(size: tuple[int, int], noise: bool = False, **save) -> bytes:
    if noise:
        image = Image.frombytes(
            "RGB", size, bytes((i * 7919) % 256 for i in range(size[0] * size[1] * 3))
        )
    else:
        image = Image.new("RGB", size, "red")

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95, **save)

    return out.getvalue()


class TestPreparingPhotos:
    def test_a_huge_noisy_photo_is_brought_under_the_byte_cap(self):
        prepared = prepare_photo(jpeg((4000, 3000), noise=True))

        assert len(prepared.full) <= MAX_PHOTO_BYTES
        assert max(Image.open(io.BytesIO(prepared.full)).size) <= 2048

    def test_a_small_clean_jpeg_passes_through(self):
        data = jpeg((800, 600))

        assert prepare_photo(data).full == data

    def test_location_data_is_stripped(self):
        exif = Image.Exif()
        exif[0x8825] = {1: "N", 2: (40.0, 39.0, 0.0)}
        data = jpeg((800, 600), exif=exif)

        full = prepare_photo(data).full

        assert full != data
        assert 0x8825 not in Image.open(io.BytesIO(full)).getexif()

    @pytest.mark.parametrize("kind", ["PNG", "WEBP", "HEIF", "TIFF", "GIF"])
    def test_other_formats_become_jpegs(self, kind):
        out = io.BytesIO()
        Image.new("RGB", (640, 480), "blue").save(out, format=kind)

        prepared = prepare_photo(out.getvalue())

        assert Image.open(io.BytesIO(prepared.full)).format == "JPEG"
        assert Image.open(io.BytesIO(prepared.thumbnail)).size == (280, 280)
