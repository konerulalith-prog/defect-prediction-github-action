
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor.java_metrics import analyze_java_file


def test_basic_java_metrics(tmp_path):
    source = """public class Sample {
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
    java_file = tmp_path / "Sample.java"
    java_file.write_text(source, encoding="utf-8")

    metrics = analyze_java_file(java_file)

    assert metrics["ClassCount"] == 1
    assert metrics["MethodCount"] == 2
    assert metrics["ConstructorCount"] == 0
    assert metrics["ApproxSumCyclomatic"] == 3
    assert metrics["ApproxMaxCyclomatic"] == 2
    assert metrics["ApproxCountSemicolon"] == 3
    assert metrics["ApproxCountStmtExe"] == 4


def test_comments_and_strings(tmp_path):
    source = """public class Comments {
    // This is a comment
    String text = "hello; world";
    /* A block comment */
    int count = 1;
}
"""
    java_file = tmp_path / "Comments.java"
    java_file.write_text(source, encoding="utf-8")

    metrics = analyze_java_file(java_file)

    # Semicolons inside strings must not be counted.
    assert metrics["ApproxCountSemicolon"] == 2

    # Blank lines are counted separately.
    assert metrics["CountLineBlank"] == 0


def test_loop_complexity(tmp_path):
    source = """public class Loops {
    public void run() {
        for (int i = 0; i < 10; i++) {
            if (i > 5) {
                break;
            }
        }
    }
}
"""
    java_file = tmp_path / "Loops.java"
    java_file.write_text(source, encoding="utf-8")

    metrics = analyze_java_file(java_file)

    assert metrics["ApproxSumCyclomatic"] == 3
    assert metrics["ApproxMaxCyclomatic"] == 3
