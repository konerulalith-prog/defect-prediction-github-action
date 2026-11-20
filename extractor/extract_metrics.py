
import argparse
import csv
from pathlib import Path

import javalang


def extract_metrics(java_file):
    path = Path(java_file)
    source = path.read_text(encoding="utf-8")

    tree = javalang.parse.parse(source)
    lines = source.splitlines()

    # Initial structural metrics for testing.
    # These are NOT yet the full 26 ReLink metrics.
    method_count = sum(
        1 for _ in tree.filter(javalang.tree.MethodDeclaration)
    )

    class_count = sum(
        1 for _ in tree.filter(javalang.tree.ClassDeclaration)
    )

    return {
        "file": str(path),
        "CountLine": len(lines),
        "ClassCount": class_count,
        "MethodCount": method_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("java_file")
    parser.add_argument(
        "--output", default="data/extracted_metrics.csv"
    )
    args = parser.parse_args()

    metrics = extract_metrics(args.java_file)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print("Java metrics extracted successfully:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
