from typer.testing import CliRunner

from aic_kb.cli import app, build_string

runner = CliRunner()


def test_build_string():
    assert build_string("world", 3) == "worldworldworld"


def test_get_package_documentation():
    result = runner.invoke(app, ["get-package-documentation", "requests"])
    assert result.exit_code == 0
