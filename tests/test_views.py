import pytest

from paprika_recipes.categories import CategoryNames
from paprika_recipes.exceptions import PaprikaUserError
from paprika_recipes.ignore import IGNORE_FILENAME
from paprika_recipes.remote import RemoteRecipe
from paprika_recipes.repository import GalleryPhotoState, Repository
from paprika_recipes.views import MARKER, build_views

DINNERS = "DINNERS-UID"
DIPS = "DIPS-UID"


@pytest.fixture
def repository(tmp_path) -> Repository:
    repository = Repository.initialize(tmp_path)
    (tmp_path / IGNORE_FILENAME).write_text("views/\nbrowse/\n", encoding="utf-8")
    repository.update_categories(CategoryNames({DINNERS: "Dinners", DIPS: "Dips"}))

    return repository


def add(repository: Repository, **fields) -> None:
    repository.store(RemoteRecipe(**fields))


def tree(path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.name != MARKER)


class TestRecipeFolders:
    def test_every_category_and_fixed_view_gets_a_folder(self, repository):
        views = build_views(repository).path

        assert sorted(p.name for p in views.iterdir() if p.is_dir()) == [
            "Dinners",
            "Dips",
            "Favorites",
            "Pinned",
            "Top Rated",
            "Uncategorized",
        ]

    def test_a_recipe_appears_in_each_of_its_categories(self, repository):
        add(repository, name="Hummus", categories=[DINNERS, DIPS])

        views = build_views(repository).path

        for folder in ("Dinners", "Dips"):
            link = views / folder / "Hummus.md"
            assert link.is_symlink()
            assert link.resolve() == (repository.root / "Hummus.md").resolve()

    def test_flags_and_ratings_have_their_own_folders(self, repository):
        add(repository, name="Stew", rating=5, on_favorites=True, is_pinned=True)
        add(repository, name="Toast")

        views = build_views(repository).path

        assert (views / "Favorites" / "Stew.md").is_symlink()
        assert (views / "Pinned" / "Stew.md").is_symlink()
        assert (views / "Top Rated" / "5★ Stew.md").is_symlink()
        assert (views / "Uncategorized" / "Toast.md").is_symlink()
        assert not (views / "Top Rated" / "0★ Toast.md").exists()

    def test_each_folder_links_the_attachments(self, repository):
        repository.attachments_dir.mkdir()

        views = build_views(repository).path

        assert (views / "Dinners" / "attachments").resolve() == (
            repository.attachments_dir.resolve()
        )

    def test_a_nested_output_still_links_correctly(self, repository):
        add(repository, name="Stew", categories=[DINNERS])

        views = build_views(repository, repository.root / "browse" / "all").path

        assert (views / "Dinners" / "Stew.md").read_text(encoding="utf-8")


class TestPhotoFolders:
    def give_gallery(self, repository, uid, *entries):
        for filename, caption in entries:
            repository.write_gallery_photo(
                uid, filename, GalleryPhotoState(uid=filename, name=caption), b"jpg"
            )

    def test_photos_are_numbered_and_captioned(self, repository):
        add(
            repository,
            uid="T",
            name="Tzatziki",
            gallery=[["a.jpg", "Tzaziki"], ["b.jpg", "Original Recipe"]],
        )
        self.give_gallery(repository, "T", ("a.jpg", "Tzaziki"), ("b.jpg", "x"))

        views = build_views(repository).path

        assert tree(views / "Photos") == [
            "Tzatziki",
            "Tzatziki/1 Tzaziki.jpg",
            "Tzatziki/2 Original Recipe.jpg",
        ]
        assert (views / "Photos" / "Tzatziki" / "1 Tzaziki.jpg").read_bytes() == b"jpg"

    def test_number_only_captions_are_left_off(self, repository):
        add(repository, uid="M", name="Meatballs", gallery=[["a.jpg", "1"]])
        self.give_gallery(repository, "M", ("a.jpg", "1"))

        views = build_views(repository).path

        assert tree(views / "Photos" / "Meatballs") == ["1.jpg"]

    def test_a_recipe_without_a_gallery_shows_its_thumbnail(self, repository):
        add(repository, uid="C", name="Chili", photo="t.jpg")
        repository.write_photo("C", "t.jpg", "hash", b"thumb")

        views = build_views(repository).path

        assert (views / "Photos" / "Chili" / "thumbnail.jpg").read_bytes() == b"thumb"

    def test_a_category_called_photos_keeps_its_name(self, repository):
        repository.update_categories(CategoryNames({DINNERS: "Photos"}))
        add(repository, uid="C", name="Chili", photo="t.jpg", categories=[DINNERS])
        repository.write_photo("C", "t.jpg", "hash", b"thumb")

        views = build_views(repository).path

        assert (views / "Photos" / "Chili.md").is_symlink()
        assert (views / "Photos (gallery)" / "Chili" / "thumbnail.jpg").exists()


class TestSafety:
    def test_rebuilding_drops_what_is_gone(self, repository):
        add(repository, name="Stew")
        build_views(repository)
        (repository.root / "Stew.md").unlink()

        views = build_views(repository).path

        assert not (views / "Uncategorized" / "Stew.md").exists()

    def test_refuses_an_output_the_ignore_file_does_not_exclude(self, repository):
        with pytest.raises(PaprikaUserError, match="paprikaignore"):
            build_views(repository, repository.root / "elsewhere")

    def test_refuses_to_replace_a_folder_it_did_not_make(self, repository):
        (repository.root / "views").mkdir()
        (repository.root / "views" / "mine.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(PaprikaUserError, match="not made by this command"):
            build_views(repository)

        assert (repository.root / "views" / "mine.txt").exists()

    def test_refuses_an_output_outside_the_directory(self, repository, tmp_path):
        with pytest.raises(PaprikaUserError, match="not inside"):
            build_views(repository, tmp_path.parent / "out")

    def test_the_views_are_not_read_as_recipes(self, repository):
        add(repository, name="Stew", categories=[DINNERS])

        build_views(repository)

        assert [entry.name for entry in repository.status()] == ["Stew"]
