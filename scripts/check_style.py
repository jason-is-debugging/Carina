"""Style and size verification for Carina's Python source tree.

Walks ``src/`` and ``tests/`` (plus any other directories given on the
command line), parses every ``.py`` file, and verifies:

    * no file exceeds 300 non-blank / non-comment lines
    * no function exceeds 50 lines (excluding the docstring)
    * no class exceeds 200 lines (excluding comments/docstrings)

Outputs ``OK`` when everything is within bounds; otherwise prints the
violations and exits with a non-zero status.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "scripts", REPO_ROOT / "configs"]

MAX_FILE_LINES = 300
MAX_FUNCTION_LINES = 50
MAX_CLASS_LINES = 200


def _stripped_lines(source: str) -> int:
    """Count code lines (non-blank, non-comment) in ``source``."""
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def _function_body_lines(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count physical lines covered by ``func`` excluding the docstring."""
    if not func.body:
        return max(0, (func.end_lineno or func.lineno) - func.lineno + 1)
    body_start = func.body[0].lineno
    first = func.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body_start = first.end_lineno + 1
    return max(0, (func.end_lineno or body_start) - body_start + 1)


def _class_metrics(cls: ast.ClassDef) -> int:
    """Count lines spanned by statements inside ``cls`` other than docstrings."""
    last = cls.lineno
    for stmt in cls.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        last = max(last, stmt.end_lineno or stmt.lineno)
    return max(0, last - cls.lineno + 1)


def _check_file(path: Path) -> list[str]:
    """Return a list of violations found in ``path``."""
    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    file_lines = _stripped_lines(source)
    if file_lines > MAX_FILE_LINES:
        violations.append(
            f"{path}: file has {file_lines} non-blank lines (limit {MAX_FILE_LINES})"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        violations.append(f"{path}: syntax error {exc}")
        return violations
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_lines = _function_body_lines(node)
            if body_lines > MAX_FUNCTION_LINES:
                violations.append(
                    f"{path}:{node.lineno} function {node.name!r} has {body_lines} lines "
                    f"(limit {MAX_FUNCTION_LINES})"
                )
        elif isinstance(node, ast.ClassDef):
            cls_lines = _class_metrics(node)
            if cls_lines > MAX_CLASS_LINES:
                violations.append(
                    f"{path}:{node.lineno} class {node.name!r} has {cls_lines} lines "
                    f"(limit {MAX_CLASS_LINES})"
                )
    return violations


def _iter_python(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            for sub in root.rglob("*.py"):
                if "__pycache__" in sub.parts:
                    continue
                files.append(sub)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Carina style constraints.")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_DIRS)
    args = parser.parse_args(argv)
    files = _iter_python(args.paths)
    if not files:
        print("No Python files found.")
        return 1
    violations: list[str] = []
    for path in files:
        violations.extend(_check_file(path))
    if violations:
        print("Style violations:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(
        f"OK — checked {len(files)} files (file≤{MAX_FILE_LINES}, "
        f"func≤{MAX_FUNCTION_LINES}, class≤{MAX_CLASS_LINES})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
