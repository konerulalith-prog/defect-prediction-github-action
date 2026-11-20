
import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extractor.relink_metrics import write_outputs
from model.predict import predict_file_metrics
from model.explain import explain_predictions

WARNING = (
    "Experimental approximate metrics only. "
    "Not validated as ReLink-equivalent."
)


def analyze_file(java_file, output_dir):
    java_file = Path(java_file)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp:
        csv_path = Path(temp) / "metrics.csv"

        _, _, audit = write_outputs(java_file, csv_path)
        prediction = predict_file_metrics(csv_path).iloc[0]

        shap_output = io.StringIO()
        with contextlib.redirect_stdout(shap_output):
            explain_predictions(csv_path)

        return {
            "file": java_file.as_posix(),
            "defect_probability": float(
                prediction["defect_probability"]
            ),
            "predicted_class": int(
                prediction["predicted_class"]
            ),
            "risk": str(prediction["risk"]),
            "unavailable_metrics": audit["unavailable"],
            "shap_explanation": shap_output.getvalue(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--output-dir", default="reports/pr_analysis"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for filename in dict.fromkeys(args.files):
        path = Path(filename)

        if path.suffix.lower() != ".java" or not path.is_file():
            continue

        try:
            results.append(analyze_file(path, output_dir))
        except Exception as exc:
            errors.append({
                "file": filename,
                "error": str(exc) or type(exc).__name__,
            })

    report = {
        "warning": WARNING,
        "files_analyzed": len(results),
        "results": results,
        "errors": errors,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "<!-- java-defect-prediction-report -->",
        "## Java Defect Prediction",
        "",
        f"> **Experimental:** {WARNING}",
        "",
        "| Java file | Defect probability | Risk |",
        "|---|---:|---|",
    ]

    for item in results:
        safe_name = item["file"].replace("|", "\\|")
        lines.append(
            f"| `{safe_name}` | "
            f"{item['defect_probability']:.2%} | "
            f"{item['risk']} |"
        )

    if not results:
        lines.append("| No Java files analyzed | — | — |")

    for item in results:
        lines.extend([
            "",
            f"### `{item['file']}`",
            "",
            "**SHAP explanation**",
            "",
            "```text",
            item["shap_explanation"].strip(),
            "```",
            "",
            "_SHAP values describe model-output contributions, "
            "not percentage-point changes in probability._",
        ])

    if errors:
        lines.extend(["", "### Analysis errors"])
        for error in errors:
            lines.append(
                f"- `{error['file']}`: {error['error']}"
            )

    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Analyzed {len(results)} Java files.")
    print(f"Errors: {len(errors)}")
    print(f"Reports saved to {output_dir}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
