
"""Approximate Java metrics using the 26 ReLink feature names.

IMPORTANT:
These measurements are experimental approximations, not validated
SciTools Understand / ReLink-equivalent measurements.

The essential-complexity estimator only handles a limited subset of
structured Java control flow. Unsupported cases remain unavailable.

Run:
    python -m extractor.relink_metrics tests/Sample.java
"""

import argparse
import csv
import json
from pathlib import Path

import javalang
from javalang import tree as jt

from .java_metrics import analyze_java_file


FEATURES = (
    "AvgCyclomatic",
    "AvgCyclomaticModified",
    "AvgCyclomaticStrict",
    "AvgEssential",
    "AvgLine",
    "AvgLineBlank",
    "AvgLineCode",
    "AvgLineComment",
    "CountLine",
    "CountLineBlank",
    "CountLineCode",
    "CountLineCodeDecl",
    "CountLineCodeExe",
    "CountLineComment",
    "CountSemicolon",
    "CountStmt",
    "CountStmtDecl",
    "CountStmtExe",
    "MaxCyclomatic",
    "MaxCyclomaticModified",
    "MaxCyclomaticStrict",
    "RatioCommentToCode",
    "SumCyclomatic",
    "SumCyclomaticModified",
    "SumCyclomaticStrict",
    "SumEssential",
)


APPROXIMATE = {
    "CountLine": "CountLine",
    "CountLineBlank": "CountLineBlank",
    "CountLineCode": "ApproxCountLineCode",
    "CountLineComment": "ApproxCountLineCommentOnly",
    "CountSemicolon": "ApproxCountSemicolon",
    "CountStmt": "ApproxCountStmt",
    "CountStmtDecl": "ApproxCountStmtDecl",
    "CountStmtExe": "ApproxCountStmtExe",
    "AvgCyclomatic": "ApproxAvgCyclomatic",
    "MaxCyclomatic": "ApproxMaxCyclomatic",
    "SumCyclomatic": "ApproxSumCyclomatic",
    "RatioCommentToCode": "ApproxRatioCommentToCode",
}


def comment_line_numbers(source):
    """Find lines containing comments using approximate lexical scanning."""
    result = set()
    state = "code"
    line = 1
    i = 0

    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if ch == "\n":
            if state == "line_comment":
                state = "code"
            line += 1
            i += 1
            continue

        if state == "code":
            if ch == "/" and nxt == "/":
                result.add(line)
                state = "line_comment"
                i += 2
                continue

            if ch == "/" and nxt == "*":
                result.add(line)
                state = "block_comment"
                i += 2
                continue

            if source.startswith('"""', i):
                state = "text_block"
                i += 3
                continue

            if ch == '"':
                state = "string"
            elif ch == "'":
                state = "character"

        elif state == "line_comment":
            result.add(line)

        elif state == "block_comment":
            result.add(line)

            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue

        elif state in ("string", "character"):
            if ch == "\\":
                i += 2
                continue

            if (
                (state == "string" and ch == '"')
                or (state == "character" and ch == "'")
            ):
                state = "code"

        elif state == "text_block":
            if source.startswith('"""', i):
                state = "code"
                i += 3
                continue

        i += 1

    return result


def method_ranges(tokens, tree):
    """Estimate method and constructor source ranges by matching braces."""
    methods = [
        node
        for _, node in tree
        if isinstance(
            node,
            (jt.MethodDeclaration, jt.ConstructorDeclaration),
        )
        and node.body is not None
        and node.position is not None
    ]

    ranges = []

    for method in methods:
        start = method.position.line

        opening = None

        for index, token in enumerate(tokens):
            if token.position is None:
                continue

            if token.position.line < start:
                continue

            if token.value == "{":
                opening = index
                break

        if opening is None:
            continue

        depth = 0
        end = None
        closing = None

        for index in range(opening, len(tokens)):
            token = tokens[index]

            if token.value == "{":
                depth += 1

            elif token.value == "}":
                depth -= 1

                if depth == 0:
                    end = token.position.line
                    closing = index
                    break

        if end is not None:
            ranges.append(
                (method, start, end, opening, closing)
            )

    return ranges


def method_complexity(method, method_tokens):
    """Calculate experimental cyclomatic complexity variants."""
    decisions = (
        jt.IfStatement,
        jt.ForStatement,
        jt.WhileStatement,
        jt.DoStatement,
        jt.CatchClause,
    )

    base = 1
    switch_count = 0
    switch_cases = 0

    for _, node in method:
        if isinstance(node, decisions):
            base += 1

        elif isinstance(node, jt.SwitchStatement):
            switch_count += 1

        elif isinstance(node, jt.SwitchStatementCase):
            if node.case:
                switch_cases += 1

    # These are experimental definitions.
    modified = base + switch_count
    ordinary = base + switch_cases

    boolean_ops = sum(
        token.value in ("&&", "||")
        for token in method_tokens
    )

    strict = ordinary + boolean_ops

    return ordinary, modified, strict


def additional_metrics(java_file):
    """Calculate 12 additional approximate Java metrics."""
    source = Path(java_file).read_text(encoding="utf-8")

    lines = source.splitlines()
    tree = javalang.parse.parse(source)
    tokens = list(javalang.tokenizer.tokenize(source))

    code_lines = {
        token.position.line
        for token in tokens
        if token.position is not None
    }

    comment_lines = comment_line_numbers(source)
    ranges = method_ranges(tokens, tree)

    declaration_lines = set()

    declaration_types = (
        jt.ClassDeclaration,
        jt.InterfaceDeclaration,
        jt.EnumDeclaration,
        jt.MethodDeclaration,
        jt.ConstructorDeclaration,
        jt.FieldDeclaration,
        jt.LocalVariableDeclaration,
    )

    for _, node in tree:
        if isinstance(node, declaration_types):
            if node.position is not None:
                declaration_lines.add(node.position.line)

    declaration_lines &= code_lines
    executable_lines = code_lines - declaration_lines

    totals = []
    blank_counts = []
    code_counts = []
    comment_counts = []

    modified_scores = []
    strict_scores = []

    for method, start, end, opening, closing in ranges:
        method_lines = set(range(start, end + 1))

        totals.append(end - start + 1)

        blank_counts.append(
            sum(
                not lines[line_number - 1].strip()
                for line_number in method_lines
                if 1 <= line_number <= len(lines)
            )
        )

        code_counts.append(
            len(method_lines & code_lines)
        )

        comment_counts.append(
            len(method_lines & comment_lines)
        )

        method_tokens = tokens[opening:closing + 1]

        ordinary, modified, strict = method_complexity(
            method, method_tokens
        )

        modified_scores.append(modified)
        strict_scores.append(strict)

    def average(values):
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    return {
        "AvgLine": average(totals),
        "AvgLineBlank": average(blank_counts),
        "AvgLineCode": average(code_counts),
        "AvgLineComment": average(comment_counts),
        "CountLineCodeDecl": len(declaration_lines),
        "CountLineCodeExe": len(executable_lines),
        "AvgCyclomaticModified": average(modified_scores),
        "MaxCyclomaticModified": max(
            modified_scores, default=0
        ),
        "SumCyclomaticModified": sum(modified_scores),
        "AvgCyclomaticStrict": average(strict_scores),
        "MaxCyclomaticStrict": max(
            strict_scores, default=0
        ),
        "SumCyclomaticStrict": sum(strict_scores),
    }


def approximate_essential_complexity(java_file):
    """Estimate essential complexity for a limited structured subset.

    This is NOT a full control-flow-graph implementation.

    For supported structured methods, estimated essential complexity
    is 1 per method. Methods with labeled jumps, switch statements,
    or other unsupported control-flow constructs are left unavailable.
    """
    source = Path(java_file).read_text(encoding="utf-8")
    tree = javalang.parse.parse(source)

    methods = [
        node
        for _, node in tree
        if isinstance(
            node,
            (jt.MethodDeclaration, jt.ConstructorDeclaration),
        )
        and node.body is not None
    ]

    if not methods:
        return {
            "AvgEssential": None,
            "SumEssential": None,
        }

    unsupported_types = (
        jt.SwitchStatement,
        jt.TryStatement,
        jt.SynchronizedStatement,
    )

    for method in methods:
        for _, node in method:
            if isinstance(node, unsupported_types):
                return {
                    "AvgEssential": None,
                    "SumEssential": None,
                }

            if isinstance(
                node,
                (jt.BreakStatement, jt.ContinueStatement),
            ):
                # javalang represents the target label as "goto".
                if getattr(node, "goto", None) is not None:
                    return {
                        "AvgEssential": None,
                        "SumEssential": None,
                    }

            # Conservative treatment of explicit labels.
            if getattr(node, "label", None) is not None:
                return {
                    "AvgEssential": None,
                    "SumEssential": None,
                }

    return {
        "AvgEssential": 1.0,
        "SumEssential": len(methods),
    }


def extract_relink_row(java_file):
    original = analyze_java_file(java_file)

    additional = additional_metrics(java_file)

    additional.update(
        approximate_essential_complexity(java_file)
    )

    row = {
        feature: None
        for feature in FEATURES
    }

    provenance = {}

    for feature in FEATURES:
        source_key = APPROXIMATE.get(feature)

        if source_key is not None:
            value = original.get(source_key)
            source_name = source_key

        elif feature in additional:
            value = additional[feature]

            if feature in ("AvgEssential", "SumEssential"):
                source_name = "limited_structured_estimate"
            else:
                source_name = "experimental_java_ast"

        else:
            value = None
            source_name = None

        row[feature] = value

        if value is not None:
            provenance[feature] = {
                "status": "approximate",
                "source": source_name,
            }
        else:
            provenance[feature] = {
                "status": "unavailable",
                "source": None,
            }

    return row, provenance


def write_outputs(java_file, csv_path):
    csv_path = Path(csv_path)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row, provenance = extract_relink_row(java_file)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FEATURES,
        )

        writer.writeheader()
        writer.writerow(row)

    audit_path = csv_path.with_suffix(".audit.json")

    audit = {
        "java_file": str(java_file),
        "warning": (
            "Experimental approximate metrics only. "
            "Not validated as ReLink-equivalent."
        ),
        "available_approximate": [
            name
            for name, info in provenance.items()
            if info["status"] == "approximate"
        ],
        "unavailable": [
            name
            for name, info in provenance.items()
            if info["status"] == "unavailable"
        ],
        "provenance": provenance,
    }

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    return csv_path, audit_path, audit


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument("java_file")

    parser.add_argument(
        "--csv",
        default="reports/relink_metrics.csv",
    )

    args = parser.parse_args()

    csv_path, audit_path, audit = write_outputs(
        args.java_file,
        args.csv,
    )

    print("CSV:", csv_path)
    print("Audit:", audit_path)

    print(
        "Approximate:",
        len(audit["available_approximate"]),
    )

    print(
        "Unavailable:",
        len(audit["unavailable"]),
    )

    print(
        "WARNING: This is not a validated "
        "ReLink-equivalent extractor."
    )


if __name__ == "__main__":
    main()
