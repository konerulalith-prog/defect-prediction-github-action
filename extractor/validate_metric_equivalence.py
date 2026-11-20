
"""Validate Java metrics against the model schema and a trusted reference."""

import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_METADATA = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "saved_model"
    / "model_metadata.json"
)


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(f"No CSV header: {path}")

        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Duplicate column names: {path}")

        return reader.fieldnames, list(reader)


def audit(extracted, metadata, reference=None, key="file", tolerance=0.01):
    features = json.loads(
        Path(metadata).read_text(encoding="utf-8")
    )["features"]

    if len(features) != 26 or len(set(features)) != 26:
        raise ValueError("Expected 26 unique model features")

    cols, rows = load_csv(extracted)

    if not rows:
        raise ValueError("No extracted rows")

    missing = [f for f in features if f not in cols]

    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))

    issues = []

    for index, row in enumerate(rows, 2):
        values = {}

        for feature in features:
            try:
                number = float(row[feature])

                if not math.isfinite(number):
                    raise ValueError("nonfinite")

                values[feature] = number

            except (TypeError, ValueError):
                issues.append(
                    f"row {index}: invalid {feature}: {row.get(feature)!r}"
                )

        if len(values) != len(features):
            continue

        for feature, value in values.items():
            if value < 0:
                issues.append(f"row {index}: negative {feature}")

        for variant in ("", "Modified", "Strict"):
            avg = values["AvgCyclomatic" + variant]
            maximum = values["MaxCyclomatic" + variant]
            total = values["SumCyclomatic" + variant]

            if avg > maximum + 1e-8:
                issues.append(
                    f"row {index}: average exceeds maximum for {variant}"
                )

            if total + 1e-8 < maximum:
                issues.append(
                    f"row {index}: sum below maximum for {variant}"
                )

        if values["CountStmt"] != (
            values["CountStmtDecl"] + values["CountStmtExe"]
        ):
            issues.append(f"row {index}: statement subtotal mismatch")

        if values["CountLineBlank"] > values["CountLine"]:
            issues.append(f"row {index}: blank lines exceed total lines")

    result = {
        "rows": len(rows),
        "features": len(features),
        "schema_valid": True,
        "internal_consistency_passed": not issues,
        "issues": issues,
        "reference_comparison": "NOT_RUN",
        "semantic_equivalence_validated": False,
    }

    if reference:
        rcols, rrows = load_csv(reference)

        if key not in cols or key not in rcols:
            raise ValueError(
                f"Both CSVs must have matching key column {key!r}"
            )

        if any(f not in rcols for f in features):
            raise ValueError("Reference missing features")

        if (
            len({row[key] for row in rows}) != len(rows)
            or len({row[key] for row in rrows}) != len(rrows)
        ):
            raise ValueError("Duplicate file keys")

        ref_by_key = {row[key]: row for row in rrows}
        differences = []
        matched = 0

        for row in rows:
            if row[key] not in ref_by_key:
                issues.append(f"No reference match for {row[key]}")
                continue

            matched += 1
            ref = ref_by_key[row[key]]

            for feature in features:
                try:
                    observed = float(row[feature])
                    expected = float(ref[feature])

                    if not (
                        math.isfinite(observed)
                        and math.isfinite(expected)
                    ):
                        raise ValueError("nonfinite")

                    if not math.isclose(
                        observed,
                        expected,
                        rel_tol=tolerance,
                        abs_tol=tolerance,
                    ):
                        differences.append({
                            "file": row[key],
                            "metric": feature,
                            "extracted": observed,
                            "reference": expected,
                        })

                except (TypeError, ValueError):
                    differences.append({
                        "file": row[key],
                        "metric": feature,
                        "error": "non-numeric comparison",
                    })

        result.update(
            reference_comparison="COMPLETED",
            matched_files=matched,
            metric_comparisons=matched * len(features),
            differing_values=len(differences),
            differences=differences[:100],
            semantic_equivalence_validated=(
                matched > 0
                and matched == len(rows)
                and not differences
                and not issues
            ),
        )

        result["issues"] = issues
        result["internal_consistency_passed"] = not issues

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit Java metrics against model metadata"
    )

    parser.add_argument("extracted")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--reference")
    parser.add_argument("--key", default="file")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output", default="reports/metric_audit.json")

    args = parser.parse_args()

    try:
        result = audit(
            args.extracted,
            args.metadata,
            args.reference,
            args.key,
            args.tolerance,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"AUDIT FAILED: {exc}\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"Wrote {output}")

    if not result["internal_consistency_passed"]:
        parser.exit(1, "Internal consistency check failed\n")

    if not result["semantic_equivalence_validated"]:
        print(
            "WARNING: Semantic equivalence NOT validated. "
            "Do not label scores as calibrated ReLink predictions."
        )


if __name__ == "__main__":
    main()
