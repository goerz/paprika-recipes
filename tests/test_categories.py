import pytest

from paprika_recipes.categories import CategoryNames
from paprika_recipes.markdown import DocumentFormat, render_recipe
from paprika_recipes.remote import RemoteRecipe
from paprika_recipes.repository import Repository, Status

DINNERS = "87FCDB99-1A0C-42BB-B436-722150E8140B"
DIPS = "4132577F-8708-4BB8-A981-95E3070C9F1F"
GONE = "00000000-0000-0000-0000-000000000000"

NAMES = CategoryNames({DINNERS: "Dinners", DIPS: "Dips"})


class TestCategoryNames:
    def test_writes_a_known_uid_as_its_name(self):
        assert NAMES.display(DINNERS) == "Dinners"

    def test_reads_a_name_back_as_its_uid(self):
        assert NAMES.resolve("Dinners") == DINNERS

    def test_still_accepts_a_uid(self):
        assert NAMES.resolve(DIPS) == DIPS

    def test_writes_an_unknown_uid_as_itself(self):
        assert NAMES.display(GONE) == GONE

    def test_keeps_an_unknown_name_as_typed(self):
        assert NAMES.resolve("Soups") == "Soups"

    def test_a_shared_name_falls_back_to_uids(self):
        """Two nested categories can share a name; neither can be read back."""
        names = CategoryNames({DINNERS: "Quick", DIPS: "Quick"})

        assert names.display(DINNERS) == DINNERS
        assert names.resolve("Quick") == "Quick"

    def test_reports_what_names_no_category(self):
        assert NAMES.unknown([DINNERS, "Soups"]) == ["Soups"]

    def test_reads_the_api_response_skipping_deleted_ones(self):
        names = CategoryNames.from_api(
            [
                {"uid": DINNERS, "name": "Dinners", "parent_uid": None},
                {"uid": DIPS, "name": "Dips", "deleted": True},
            ]
        )

        assert names == CategoryNames({DINNERS: "Dinners"})

    def test_round_trips_through_a_file(self, tmp_path):
        path = tmp_path / "categories.json"
        NAMES.save(path)

        assert CategoryNames.load(path) == NAMES

    def test_a_missing_or_broken_file_knows_nothing(self, tmp_path):
        assert not CategoryNames.load(tmp_path / "missing.json")

        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")

        assert not CategoryNames.load(broken)


class TestDocumentFormat:
    FORMAT = DocumentFormat(categories=NAMES)

    def test_renders_names(self):
        rendered = self.FORMAT.render(RemoteRecipe(name="Stew", categories=[DINNERS]))

        assert "- Dinners\n" in rendered
        assert DINNERS not in rendered

    def test_parses_names_back_into_uids(self):
        recipe = RemoteRecipe(name="Stew", categories=[DINNERS, DIPS])
        parsed = self.FORMAT.parse_recipe(self.FORMAT.render(recipe), RemoteRecipe)

        assert parsed.categories == [DINNERS, DIPS]

    def test_round_trips_without_loss(self):
        recipe = RemoteRecipe(name="Stew", categories=[DINNERS, GONE, "Soups"])

        assert self.FORMAT.find_lossy_fields(recipe) == []

    def test_respects_a_frontmatter_prefix(self):
        prefixed = DocumentFormat("paprika_", NAMES)
        rendered = prefixed.render(RemoteRecipe(name="Stew", categories=[DINNERS]))

        assert "paprika_categories:\n- Dinners\n" in rendered
        assert prefixed.parse_recipe(rendered, RemoteRecipe).categories == [DINNERS]

    def test_without_names_values_pass_through(self):
        """An archive already holds names, and has no list to look them up in."""
        rendered = render_recipe(RemoteRecipe(name="Stew", categories=["Dinners"]))

        assert "- Dinners\n" in rendered


@pytest.fixture
def repository(tmp_path) -> Repository:
    repository = Repository.initialize(tmp_path)
    repository.update_categories(NAMES)

    return repository


def set_categories(path, *names: str) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    start = lines.index("categories:")
    end = start + 1

    while lines[end].startswith("- "):
        end += 1

    lines[start + 1 : end] = [f"- {name}" for name in names]
    path.write_text("\n".join(lines), encoding="utf-8")


class TestInARepository:
    def test_a_pulled_recipe_shows_names_and_reads_as_unchanged(self, repository):
        path = repository.store(RemoteRecipe(name="Stew", categories=[DINNERS]))

        assert "- Dinners\n" in path.read_text(encoding="utf-8")
        assert [entry.status for entry in repository.status()] == [Status.UNCHANGED]

    def test_adding_a_category_by_name_is_an_edit_in_uids(self, repository):
        path = repository.store(RemoteRecipe(name="Stew", categories=[DINNERS]))
        set_categories(path, "Dinners", "Dips")

        [entry] = repository.status()

        assert entry.changed_fields() == ["categories"]
        assert entry.recipe is not None
        assert entry.recipe.categories == [DINNERS, DIPS]

    def test_a_renamed_category_rewrites_the_files_naming_it(self, repository):
        path = repository.store(RemoteRecipe(name="Stew", categories=[DINNERS]))
        other = repository.store(RemoteRecipe(name="Hummus", categories=[DIPS]))
        before = other.read_text(encoding="utf-8")

        rewritten = repository.update_categories(
            CategoryNames({DINNERS: "Suppers", DIPS: "Dips"})
        )

        assert rewritten == [path]
        assert "- Suppers\n" in path.read_text(encoding="utf-8")
        assert other.read_text(encoding="utf-8") == before
        assert {entry.status for entry in repository.status()} == {Status.UNCHANGED}

    def test_a_rename_keeps_local_edits(self, repository):
        path = repository.store(
            RemoteRecipe(name="Stew", ingredients="1 onion", categories=[DIPS])
        )
        set_categories(path, "Dinners")
        path.write_text(
            path.read_text(encoding="utf-8").replace("1 onion", "2 onions"),
            encoding="utf-8",
        )

        repository.update_categories(CategoryNames({DINNERS: "Suppers", DIPS: "Dips"}))

        [entry] = repository.status()
        assert entry.recipe is not None
        assert entry.recipe.ingredients == "2 onions"
        assert entry.recipe.categories == [DINNERS]
        assert "- Suppers\n" in path.read_text(encoding="utf-8")

    def test_a_directory_cloned_before_names_existed_is_upgraded(self, tmp_path):
        """Files written with uids are rewritten with names, and stay clean."""
        repository = Repository.initialize(tmp_path)
        path = repository.store(RemoteRecipe(name="Stew", categories=[DINNERS]))
        assert f"- {DINNERS}\n" in path.read_text(encoding="utf-8")

        repository.update_categories(NAMES)

        assert "- Dinners\n" in path.read_text(encoding="utf-8")
        assert [entry.status for entry in repository.status()] == [Status.UNCHANGED]
