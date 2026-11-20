
"""
Java metric validation tests.

Run:
    python -m pytest tests/test_metric_validation.py -v

These tests check:
- Feature completeness
- Numeric validity
- Internal consistency
- Basic cyclomatic complexity
- Unsupported control-flow handling

They do NOT establish independent ReLink equivalence.
"""

import math

import pytest

from extractor.relink_metrics import FEATURES, extract_relink_row


JAVA_CASES = {
    "simple": """
public class Simple {
    public int add(int a, int b) {
        return a + b;
    }
}
""",
    "conditions": """
public class Conditions {
    public int classify(int x) {
        if (x > 10) {
            return 2;
        } else if (x > 0) {
            return 1;
        }
        return 0;
    }
}
""",
    "loops": """
public class Loops {
    public int count(int n) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (i > 2) {
                total++;
            }
        }
        return total;
    }
}
""",
    "comments": """
public class Comments {
    // This is a comment
    public int getValue() {
        String text = "hello; world";
        /* Another comment */
        return 10;
    }
}
""",
    "nested": """
public class Outer {
    public int run() {
        return 1;
    }

    class Inner {
        public int execute(int x) {
            if (x > 0) {
                return x;
            }
            return 0;
        }
    }
}
""",
    "exceptions": """
public class Exceptions {
    public int divide(int a, int b) {
        try {
            return a / b;
        } catch (ArithmeticException e) {
            return 0;
        } finally {
            System.out.println("Finished");
        }
    }
}
""",
}


@pytest.mark.parametrize("case_name", JAVA_CASES)
def test_all_features_present(tmp_path, case_name):
    java_file = tmp_path / f"{case_name}.java"
    java_file.write_text(
        JAVA_CASES[case_name],
        encoding="utf-8",
    )

    row, provenance = extract_relink_row(java_file)

    assert len(FEATURES) == 26
    assert set(row) == set(FEATURES)
    assert set(provenance) == set(FEATURES)


@pytest.mark.parametrize("case_name", JAVA_CASES)
def test_available_metrics_are_valid(tmp_path, case_name):
    java_file = tmp_path / f"{case_name}.java"
    java_file.write_text(
        JAVA_CASES[case_name],
        encoding="utf-8",
    )

    row, provenance = extract_relink_row(java_file)

    for feature in FEATURES:
        value = row[feature]
        status = provenance[feature]["status"]

        if status == "unavailable":
            assert value is None
        else:
            assert value is not None
            assert isinstance(value, (int, float))
            assert math.isfinite(value)
            assert value >= 0


@pytest.mark.parametrize("case_name", JAVA_CASES)
def test_line_and_statement_consistency(tmp_path, case_name):
    java_file = tmp_path / f"{case_name}.java"
    java_file.write_text(
        JAVA_CASES[case_name],
        encoding="utf-8",
    )

    row, _ = extract_relink_row(java_file)

    assert row["CountLine"] >= row["CountLineBlank"]
    assert row["CountLine"] >= row["CountLineCode"]

    if all(
        row[name] is not None
        for name in (
            "CountLineCode",
            "CountLineCodeDecl",
            "CountLineCodeExe",
        )
    ):
        assert (
            row["CountLineCodeDecl"]
            + row["CountLineCodeExe"]
            == row["CountLineCode"]
        )


@pytest.mark.parametrize(
    "case_name,expected_sum,expected_max",
    [
        ("simple", 1, 1),
        ("conditions", 3, 3),
        ("loops", 3, 3),
    ],
)
def test_basic_cyclomatic_complexity(
    tmp_path,
    case_name,
    expected_sum,
    expected_max,
):
    java_file = tmp_path / f"{case_name}.java"
    java_file.write_text(
        JAVA_CASES[case_name],
        encoding="utf-8",
    )

    row, _ = extract_relink_row(java_file)

    assert row["SumCyclomatic"] == expected_sum
    assert row["MaxCyclomatic"] == expected_max


def test_extractor_marks_unsupported_switch_unavailable(tmp_path):
    java_file = tmp_path / "SwitchExample.java"

    java_file.write_text(
        """
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
""",
        encoding="utf-8",
    )

    row, provenance = extract_relink_row(java_file)

    assert row["AvgEssential"] is None
    assert row["SumEssential"] is None

    assert provenance["AvgEssential"]["status"] == "unavailable"
    assert provenance["SumEssential"]["status"] == "unavailable"
