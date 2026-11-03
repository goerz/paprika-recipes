import argparse
from pathlib import PurePosixPath

import pytest

from paprika_recipes.archive import Archive
from paprika_recipes.commands import create_archive
from paprika_recipes.exceptions import PaprikaUserError
from paprika_recipes.ignore import IGNORE_FILENAME, IgnoreRules
from paprika_recipes.markdown import render_recipe
from paprika_recipes.remote import RemoteRecipe
from paprika_recipes.repository import Repository, Status, recipe_files


def rules(*lines: str) -> IgnoreRules:
    return IgnoreRules.parse("\n".join(lines))


def ignores(rule: IgnoreRules, path: str) -> bool:
    return rule.ignores(PurePosixPath(path))


class TestPatterns:
    def test_a_plain_name_matches_at_any_depth(self):
        rule = rules("README.md")

        assert ignores(rule, "README.md")
        assert ignores(rule, "Breads/README.md")
        assert not ignores(rule, "README.md.md")

    def test_a_glob_matches_names(self):
        rule = rules("*.draft.md")

        assert ignores(rule, "Soup.draft.md")
        assert ignores(rule, "Soups/Soup.draft.md")
        assert not ignores(rule, "Soup.md")

    def test_a_directory_pattern_matches_everything_beneath_it(self):
        rule = rules("drafts/")

        assert ignores(rule, "drafts/Soup.md")
        assert ignores(rule, "Soups/drafts/deep/Soup.md")

    def test_a_directory_pattern_does_not_match_a_file_of_that_name(self):
        assert not ignores(rules("drafts.md/"), "drafts.md")

    def test_a_pattern_with_a_slash_is_anchored_to_the_top(self):
        rule = rules("/CLAUDE.md")

        assert ignores(rule, "CLAUDE.md")
        assert not ignores(rule, "Notes/CLAUDE.md")

    def test_an_anchored_star_does_not_cross_directories(self):
        rule = rules("notes/*.md")

        assert ignores(rule, "notes/a.md")
        assert not ignores(rule, "notes/deep/a.md")

    def test_a_double_star_crosses_directories(self):
        rule = rules("notes/**/*.md")

        assert ignores(rule, "notes/a.md")
        assert ignores(rule, "notes/deep/a.md")

    def test_an_anchored_directory_excludes_its_contents(self):
        assert ignores(rules("/inbox"), "inbox/Soup.md")

    def test_blank_lines_and_comments_are_skipped(self):
        rule = rules("", "# README.md", "   ")

        assert not rule
        assert not ignores(rule, "README.md")

    def test_negation_is_refused_rather_than_ignored(self):
        with pytest.raises(PaprikaUserError, match="line 2: negated"):
            rules("*.md", "!Soup.md")

    def test_a_pattern_that_names_nothing_is_refused(self):
        with pytest.raises(PaprikaUserError, match="does not name anything"):
            rules("/")

    def test_no_file_means_nothing_is_ignored(self, tmp_path):
        assert not IgnoreRules.load(tmp_path)


@pytest.fixture
def repository(tmp_path) -> Repository:
    return Repository.initialize(tmp_path)


def write_ignore(repository: Repository, *lines: str) -> None:
    (repository.root / IGNORE_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_notes(repository: Repository, name: str = "CLAUDE.md") -> None:
    path = repository.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Instructions\n\nNot a recipe.\n", encoding="utf-8")


class TestInARepository:
    def test_an_ignored_file_is_not_a_recipe(self, repository):
        write_ignore(repository, "CLAUDE.md", "README.md")
        write_notes(repository, "CLAUDE.md")
        write_notes(repository, "README.md")

        assert repository.status() == []

    def test_without_the_rule_the_same_file_is_read_as_a_recipe(self, repository):
        (repository.root / "CLAUDE.md").write_text(
            "---\n---\n\n# Instructions\n", encoding="utf-8"
        )

        assert [entry.status for entry in repository.status()] == [Status.ADDED]

    def test_recipes_beside_an_ignored_file_are_still_found(self, repository):
        write_ignore(repository, "CLAUDE.md")
        write_notes(repository)
        repository.store(RemoteRecipe(name="Soup"))

        assert [entry.name for entry in repository.status()] == ["Soup"]

    def test_a_recipe_is_never_written_to_an_ignored_name(self, repository):
        write_ignore(repository, "/README.md")

        path = repository.store(RemoteRecipe(name="README"))

        assert path.name != "README.md"
        assert [entry.status for entry in repository.status()] == [Status.UNCHANGED]

    def test_ignoring_a_tracked_recipe_is_refused(self, repository):
        """It would otherwise look deleted, and push would trash it."""
        path = repository.store(RemoteRecipe(name="Soup"))
        write_ignore(repository, path.name)

        with pytest.raises(PaprikaUserError, match="excludes it"):
            repository.status()

    def test_an_unreadable_ignored_file_does_not_stop_a_deletion(self, repository):
        path = repository.store(RemoteRecipe(name="Soup"))
        path.unlink()
        write_ignore(repository, "CLAUDE.md")
        write_notes(repository)

        assert [entry.status for entry in repository.status()] == [Status.DELETED]

    def test_recipe_files_honours_the_directorys_own_rules(self, tmp_path):
        (tmp_path / IGNORE_FILENAME).write_text("drafts/\n", encoding="utf-8")
        (tmp_path / "drafts").mkdir()
        (tmp_path / "drafts" / "Soup.md").write_text("draft", encoding="utf-8")
        (tmp_path / "Stew.md").write_text("stew", encoding="utf-8")

        assert [path.name for path in recipe_files(tmp_path)] == ["Stew.md"]


class TestArchives:
    def test_create_archive_skips_ignored_files(self, tmp_path):
        root = tmp_path / "recipes"
        root.mkdir()
        (root / "Soup.md").write_text(
            render_recipe(RemoteRecipe(name="Soup", uid="ABC")), encoding="utf-8"
        )
        (root / "README.md").write_text("Not a recipe.\n", encoding="utf-8")
        (root / IGNORE_FILENAME).write_text("README.md\n", encoding="utf-8")

        archive_path = tmp_path / "out.paprikarecipes"
        create_archive.Command(
            argparse.Namespace(export_path=root, archive_path=archive_path)
        ).handle()

        with open(archive_path, "rb") as inf:
            assert Archive.from_file(inf).count() == 1
