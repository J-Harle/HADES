from pathlib import Path

import hades


def test_relative_input_is_resolved_from_working_directory(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["hades", "--input", "molecules.csv"])

    hades.main()

    output = capsys.readouterr().out
    assert f"Input: {Path(tmp_path, 'molecules.csv')}" in output
