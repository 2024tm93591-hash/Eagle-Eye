"""Formats evaluation results as plain text.

The Analytics page and scripts/evaluate.py both use this, so the numbers look the same in
the browser and in the terminal.
"""
from __future__ import annotations

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
    ("fps", "Frames per second"),
]


def format_value(key, value) -> str:
    if value is None:
        return "-"
    if key == "fps":
        return f"{value:.1f}"
    return f"{value:.3f}"


def confusion_matrix_lines(matrix, labels) -> list[str]:
    cells = [["-" if v is None else str(v) for v in row] for row in matrix]
    first_col = max(len("actual \\ predicted"), *(len(label) for label in labels))
    col_width = max(8, *(len(label) for label in labels), *(len(c) for row in cells for c in row)) + 2

    lines = ["actual \\ predicted".ljust(first_col) + "".join(label.rjust(col_width) for label in labels)]
    for label, row in zip(labels, cells):
        lines.append(label.ljust(first_col) + "".join(c.rjust(col_width) for c in row))
    return lines


def algorithm_report(metrics: dict, labels=None) -> str:
    lines = []
    for key, name in METRICS:
        if metrics.get(key) is not None:
            lines.append(f"{name:<19} {format_value(key, metrics[key])}")

    matrix = metrics.get("confusion")
    if matrix:
        lines.append("")
        lines.append("Confusion matrix")
        lines += confusion_matrix_lines(matrix, metrics.get("labels") or labels)
    return "\n".join(lines)
