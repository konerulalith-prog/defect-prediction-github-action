
import csv
import json

from extractor.relink_metrics import (
    FEATURES,
    extract_relink_row,
    write_outputs,
)


SIMPLE_JAVA = """
public class Sample {
    public int add(int a, int b) {
        return a + b;
    }

    public boolean positive(int n) {
        if (n > 0) {
            return true;
        }
        return false;
    }
}
"""


def make_java_file(tmp_path, source=SIMPLE_JAVA):
    path = tmp_path / "Sample.java"
    path.write_text(source, encoding="utf-8")
    return path


def test_all_26_metrics_are_present(tmp_path):
    path = make_java_file(tmp_path)
    row, provenance = extract_relink_row(path)

    assert len(FEATURES) == 26
    assert len(set(FEATURES)) == 26
    assert set(row) == set(FEATURES)
    assert set(provenance) == set(FEATURES)
    assert all(row[name] is not None for name in FEATURES)


def test_basic_complexity_metrics(tmp_path):
    path = make_java_file(tmp_path)
    row, _ = extract_relink_row(path)

    assert row["SumCyclomatic"] == 3
    assert row["MaxCyclomatic"] == 2
    assert row["AvgCyclomatic"] == 1.5

    assert row["AvgEssential"] == 1.0
    assert row["SumEssential"] == 2


def test_line_metrics_are_consistent(tmp_path):
    path = make_java_file(tmp_path)
    row, _ = extract_relink_row(path)

    assert row["CountLine"] >= row["CountLineCode"]
    assert row["CountLineCodeDecl"] >= 0
    assert row["CountLineCodeExe"] >= 0

    assert (
        row["CountLineCodeDecl"] + row["CountLineCodeExe"]
        == row["CountLineCode"]
    )


def test_unsupported_control_flow_is_unavailable(tmp_path):
    source = """
public class SwitchExample {
    public int check(int x) {
        switch (x) {
            case 1:
                return 10;
            default:
                return 0;
        }
    }
}
"""
    path = make_java_file(tmp_path, source)
    row, provenance = extract_relink_row(path)

    assert row["AvgEssential"] is None
    assert row["SumEssential"] is None
    assert provenance["AvgEssential"]["status"] == "unavailable"
    assert provenance["SumEssential"]["status"] == "unavailable"


def test_csv_and_audit_outputs(tmp_path):
    path = make_java_file(tmp_path)
    csv_path = tmp_path / "metrics.csv"

    output_csv, audit_path, audit = write_outputs(
        path, csv_path
    )

    assert output_csv.exists()
    assert audit_path.exists()

    with output_csv.open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames == list(FEATURES)
    assert len(rows) == 1

    saved_audit = json.loads(
        audit_path.read_text(encoding="utf-8")
    )

    assert saved_audit == audit
    assert len(audit["available_approximate"]) == 26
    assert len(audit["unavailable"]) == 0
