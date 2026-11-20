
import argparse
import json
from pathlib import Path


def format_report(input_path, output_path):
    report = json.loads(
        Path(input_path).read_text(encoding="utf-8")
    )

    lines = [
        "# AI-Powered Java Defect Prediction",
        "",
        f"**Model:** {report['model']}",
        f"**Features:** {report['feature_count']}",
        f"**Decision threshold:** "
        f"{report['decision_threshold']:.0%}",
        "",
        "> **Experimental:** Model predictions require "
        "ReLink-compatible input metrics. The current "
        "open-source Java extractor is not yet validated "
        "for this model.",
        "",
        "## Prediction Results",
        "",
        "| Sample | Defect Probability | Risk |",
        "|---|---:|---|",
    ]

    for result in report["results"]:
        lines.append(
            f"| {result['sample']} | "
            f"{result['defect_probability']:.2%} | "
            f"{result['risk']} |"
        )

    for result in report["results"]:
        lines.extend([
            "",
            f"## Sample {result['sample']}: "
            "SHAP Explanations",
            "",
            "| Feature | SHAP Value | Direction |",
            "|---|---:|---|",
        ])

        for item in result["top_features"]:
            direction = item["direction"].replace("_", " ")
            lines.append(
                f"| {item['feature']} | "
                f"{item['shap_value']:+.4f} | "
                f"{direction} |"
            )

    lines.extend([
        "",
        "_SHAP values describe contributions to the "
        "model's raw output, not direct percentage-point "
        "changes in defect probability._",
        "",
    ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"Markdown report generated: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument(
        "--output",
        default="reports/defect_report.md"
    )

    args = parser.parse_args()

    format_report(args.input_json, args.output)
