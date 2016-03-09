from click.testing import CliRunner

from shclassify.scripts.cli import cli


def test_cli_exits_if_input_file_nonexistant():
    runner = CliRunner()
    result = runner.invoke(cli, ['nopath'])
    assert 'Path "nopath" does not exist' in result.output
    assert result.exit_code != 0
