
from pathlib import Path
import argparse
import javalang
from javalang import tree as jt


def analyze_lines(source):
    lines = source.splitlines()
    blank = code = comments = 0

    for line in lines:
        if not line.strip():
            blank += 1

    # Preliminary counts; not ReLink-equivalent.
    tokens = list(javalang.tokenizer.tokenize(source))

    code_line_numbers = set(
        token.position.line
        for token in tokens
        if token.position is not None
    )

    code = len(code_line_numbers)
    comments = len(lines) - blank - code

    return {
        "CountLine": len(lines),
        "CountLineBlank": blank,
        "ApproxCountLineCode": code,
        "ApproxCountLineCommentOnly": comments,
        "ApproxRatioCommentToCode": (
            round(comments / code, 4) if code else 0
        ),
    }


def calculate_complexity(method):
    score = 1

    decisions = (
        jt.IfStatement,
        jt.ForStatement,
        jt.WhileStatement,
        jt.DoStatement,
        jt.CatchClause,
    )

    if method.body:
        for _, node in method:
            if isinstance(node, decisions):
                score += 1

    return score


def analyze_statements(tree, source):
    tokens = list(javalang.tokenizer.tokenize(source))

    # The tokenizer excludes comments and treats
    # string literals as single tokens.
    semicolons = sum(
        token.value == ";" for token in tokens
    )

    declaration_types = (
        jt.LocalVariableDeclaration,
        jt.FieldDeclaration,
    )

    declarations = sum(
        1 for _, node in tree
        if isinstance(node, declaration_types)
    )

    execution_types = (
        jt.StatementExpression,
        jt.ReturnStatement,
        jt.ThrowStatement,
        jt.BreakStatement,
        jt.ContinueStatement,
        jt.AssertStatement,
        jt.IfStatement,
        jt.ForStatement,
        jt.WhileStatement,
        jt.DoStatement,
        jt.SwitchStatement,
        jt.TryStatement,
        jt.SynchronizedStatement,
    )

    executions = sum(
        1 for _, node in tree
        if isinstance(node, execution_types)
    )

    return {
        "ApproxCountSemicolon": semicolons,
        "ApproxCountStmtDecl": declarations,
        "ApproxCountStmtExe": executions,
        "ApproxCountStmt": declarations + executions,
    }


def analyze_java_file(file_path):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    source = path.read_text(encoding="utf-8")
    tree = javalang.parse.parse(source)

    lines = analyze_lines(source)
    statements = analyze_statements(tree, source)

    classes = list(
        tree.filter(jt.ClassDeclaration)
    )
    methods = list(
        tree.filter(jt.MethodDeclaration)
    )
    constructors = list(
        tree.filter(jt.ConstructorDeclaration)
    )

    scores = [
        calculate_complexity(node)
        for _, node in methods + constructors
    ]

    return {
        "file": str(path),
        **lines,
        **statements,
        "ClassCount": len(classes),
        "MethodCount": len(methods),
        "ConstructorCount": len(constructors),
        "ApproxSumCyclomatic": sum(scores),
        "ApproxMaxCyclomatic": max(scores, default=0),
        "ApproxAvgCyclomatic": (
            round(sum(scores) / len(scores), 4)
            if scores else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("java_file")
    args = parser.parse_args()

    try:
        metrics = analyze_java_file(args.java_file)

        print("\nJava Metrics Analysis")
        print("-----------------------------")

        for name, value in metrics.items():
            print(f"{name}: {value}")

        print("-----------------------------")
        print("Analysis completed successfully.")

    except (
        FileNotFoundError,
        UnicodeError,
        javalang.parser.JavaSyntaxError,
        javalang.tokenizer.LexerError,
    ) as error:
        print(f"Analysis failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
