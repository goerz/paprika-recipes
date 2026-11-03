import argparse
from pathlib import Path

from ..command import RepositoryCommand
from ..constants import ExitCode
from ..reporting import emit_json
from ..views import build_views


class Command(RepositoryCommand):
    @classmethod
    def get_help(cls) -> str:
        return """Rebuilds a folder of symlinks for browsing the recipes by
        category, favorites, rating, and photos."""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            metavar="PATH",
            help=(
                "where to build the views, inside the recipe directory and "
                "excluded by its .paprikaignore; default: views/"
            ),
        )

    def handle(self) -> ExitCode:
        # Offline, like `status`: everything it needs is in the directory.
        report = build_views(self.repository, self.options.output)

        if self.json_output:
            emit_json(
                {
                    "path": str(report.path),
                    "recipe_links": report.recipe_links,
                    "photo_links": report.photo_links,
                    "skipped": list(report.skipped),
                }
            )
        else:
            for problem in report.skipped:
                self.console.print(f"[yellow]skipped[/yellow] {problem}")

            self.console.print(
                f"{report.recipe_links} recipe links and {report.photo_links} "
                f"photo links in {report.path}"
            )

        return ExitCode.ATTENTION if report.skipped else ExitCode.SUCCESS
