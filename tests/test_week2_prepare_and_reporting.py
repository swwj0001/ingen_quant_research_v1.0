import subprocess
import sys
from pathlib import Path

import pandas as pd

from ingen_qr.evaluation.reporting import coverage_break_section


def test_prepare_preserves_root_data_rule_and_allows_new_loader_commit(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    ignore = tmp_path / ".gitignore"
    original = "/data/\n.env\ncustom-output/\n"
    ignore.write_text(original, encoding="utf-8")
    loader = tmp_path / "src/ingen_qr/data/price_audit.py"
    loader.parent.mkdir(parents=True)
    loader.write_text("# Synthetic loader\n", encoding="utf-8")
    dataset = tmp_path / "data/example.csv"
    dataset.parent.mkdir()
    dataset.write_text("value\n1\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_week2.py"
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)
    first = ignore.read_text("utf-8")
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)
    assert ignore.read_text("utf-8") == first
    assert first.startswith(original)
    assert set(first.splitlines()) - set(original.splitlines()) == {
        "mlruns/",
        ".hypothesis/",
    }
    assert "data/" not in first.splitlines()
    assert (
        subprocess.run(
            ["git", "check-ignore", "src/ingen_qr/data/price_audit.py"],
            cwd=tmp_path,
            capture_output=True,
            check=False,
        ).returncode
        == 1
    )
    assert (
        subprocess.run(
            ["git", "check-ignore", "data/example.csv"],
            cwd=tmp_path,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    subprocess.run(
        ["git", "add", "src/ingen_qr/data/price_audit.py"], cwd=tmp_path, check=True
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert staged.stdout.strip() == "src/ingen_qr/data/price_audit.py"


def test_coverage_report_lists_every_break_row_and_column(tmp_path: Path) -> None:
    path = tmp_path / "coverage_breaks.csv"
    frame = pd.DataFrame(
        {
            "date": ["2010-03-01", "2010-04-01"],
            "feature": ["gross_margin", "beta_x_credit"],
            "coverage": [0.8, 0.3],
            "coverage_change": [0.8, -0.25],
        }
    )
    frame.to_csv(path, index=False)
    text = coverage_break_section(path)
    for column in frame.columns:
        assert column in text
    for value in ["2010-03-01", "2010-04-01", "gross_margin", "beta_x_credit", "-0.25"]:
        assert value in text


def test_empty_coverage_breaks_are_explicitly_reported(tmp_path: Path) -> None:
    path = tmp_path / "coverage_breaks.csv"
    pd.DataFrame(columns=["date", "feature", "coverage_change"]).to_csv(
        path, index=False
    )
    assert "No feature coverage changes met" in coverage_break_section(path)
