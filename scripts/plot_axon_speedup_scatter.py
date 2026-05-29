#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


TASK_COLORS = {
    "causal_lm": "#2563eb",
    "masked_lm": "#16a34a",
    "seq2seq_lm": "#dc2626",
    "unknown": "#6b7280",
}

KIND_MARKERS = {
    "dense": "circle",
    "moe": "diamond",
    "ssm": "triangle",
    "other": "square",
}

MOE_HINTS = (
    "moe",
    "mixtral",
    "gpt-oss",
    "deepseek",
    "granitemoe",
    "olmoe",
    "phimoe",
    "qwen3-moe",
    "llama4",
    "gemma-4-moe",
)
SSM_HINTS = ("mamba", "jamba", "ssm")


@dataclass(frozen=True)
class Point:
    axon: str
    checkpoint: str
    hf_time: float
    axon_time: float
    ratio: float
    task: str
    kind: str
    is_generic: bool
    top1: str
    max_abs: str


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _task_from_axon_file(axon_file: str) -> str:
    try:
        text = Path(axon_file).read_text(encoding="utf-8")
    except Exception:
        return "unknown"
    match = re.search(r'\{-#\s*TASK\s+"([^"]+)"\s*#-\}', text)
    return match.group(1) if match else "unknown"


def _model_kind(axon_file: str, checkpoint: str) -> str:
    text = f"{axon_file} {checkpoint}".lower()
    if any(hint in text for hint in SSM_HINTS):
        return "ssm"
    if any(hint in text for hint in MOE_HINTS):
        return "moe"
    # Encoder-only and encoder-decoder are still dense transformer families for
    # this high-level performance plot; task color separates them.
    return "dense"


def _point_from_row(row: dict[str, str], *, normalized_128: bool) -> Point | None:
    hf_key = "hf_time_norm128" if normalized_128 and row.get("hf_time_norm128") else "hf_time"
    axon_key = "axon_time_norm128" if normalized_128 and row.get("axon_time_norm128") else "axon_time"
    ratio_key = (
        "speed_ratio_axon_over_hf_norm128"
        if normalized_128 and row.get("speed_ratio_axon_over_hf_norm128")
        else "speed_ratio_axon_over_hf"
    )
    try:
        hf_time = float(row.get(hf_key) or "")
        axon_time = float(row.get(axon_key) or "")
    except ValueError:
        return None
    if hf_time <= 0 or axon_time <= 0:
        return None
    axon_file = str(row.get("axon") or row.get("axon_file") or "")
    checkpoint = str(row.get("checkpoint") or row.get("checkpoint_id") or "")
    if not axon_file or not checkpoint:
        return None
    ratio_raw = row.get(ratio_key) or ""
    try:
        ratio = float(ratio_raw) if ratio_raw else axon_time / hf_time
    except ValueError:
        ratio = axon_time / hf_time
    task = str(row.get("model_task") or _task_from_axon_file(axon_file))
    if task not in TASK_COLORS:
        task = "unknown"
    return Point(
        axon=axon_file,
        checkpoint=checkpoint,
        hf_time=hf_time,
        axon_time=axon_time,
        ratio=ratio,
        task=task,
        kind=_model_kind(axon_file, checkpoint),
        is_generic=Path(axon_file).name.startswith("generic-"),
        top1=str(row.get("masked_top1_eq", "")),
        max_abs=str(row.get("masked_max_abs_diff", "")),
    )


def _collect_points(log_dir: Path, *, normalized_128: bool) -> list[Point]:
    points: list[Point] = []
    if log_dir.is_file() and log_dir.suffix == ".csv":
        with log_dir.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                point = _point_from_row(row, normalized_128=normalized_128)
                if point is not None:
                    points.append(point)
        return points
    for path in sorted(log_dir.rglob("*.result.json")):
        data = _read_json(path)
        if not data:
            continue
        hf_time = data.get("hf_time")
        axon_time = data.get("axon_time")
        if not isinstance(hf_time, (int, float)) or not isinstance(axon_time, (int, float)):
            continue
        if hf_time <= 0 or axon_time <= 0:
            continue
        axon_file = str(data.get("axon_file") or "")
        checkpoint = str(data.get("checkpoint_id") or "")
        task = str(data.get("model_task") or _task_from_axon_file(axon_file))
        if task not in TASK_COLORS:
            task = "unknown"
        ratio = float(data.get("speed_ratio_axon_over_hf") or (axon_time / hf_time))
        points.append(
            Point(
                axon=axon_file,
                checkpoint=checkpoint,
                hf_time=float(hf_time),
                axon_time=float(axon_time),
                ratio=ratio,
                task=task,
                kind=_model_kind(axon_file, checkpoint),
                is_generic=Path(axon_file).name.startswith("generic-"),
                top1=str(data.get("masked_top1_eq")),
                max_abs=str(data.get("masked_max_diff", data.get("masked_max_abs_diff", ""))),
            )
        )
    return points


def _nice_log_bounds(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo <= 0:
        raise ValueError("log plot requires positive timing values")
    lo_exp = math.floor(math.log10(lo))
    hi_exp = math.ceil(math.log10(hi))
    return 10.0**lo_exp, 10.0**hi_exp


def _log_ticks(lo: float, hi: float) -> list[float]:
    start = math.floor(math.log10(lo))
    end = math.ceil(math.log10(hi))
    return [10.0**exp for exp in range(start, end + 1)]


def _fmt_time(value: float) -> str:
    if value < 1:
        return f"{value:.2g}s"
    if value < 10:
        return f"{value:.2f}s"
    return f"{value:.0f}s"


def _label(point: Point) -> str:
    axon = Path(point.axon).name
    ck = point.checkpoint.split("/")[-1]
    return f"{axon} / {ck}"


def _short_checkpoint_label(point: Point) -> str:
    return point.checkpoint.split("/")[-1]


def _dedupe_outlier_labels(points: list[Point]) -> list[Point]:
    by_checkpoint: dict[str, list[Point]] = {}
    for point in points:
        by_checkpoint.setdefault(point.checkpoint, []).append(point)
    selected: list[Point] = []
    for checkpoint_points in by_checkpoint.values():
        generic = [point for point in checkpoint_points if point.is_generic]
        candidates = generic or checkpoint_points
        selected.append(max(candidates, key=lambda point: 1.0 / point.ratio))
    return selected


def _spread_label_positions(
    labels: list[tuple[Point, float, float]],
    *,
    min_gap: float,
    top: float,
    bottom: float,
) -> dict[Point, float]:
    if not labels:
        return {}
    ordered = sorted(labels, key=lambda item: item[2])
    ys: list[float] = []
    for _, _, y in ordered:
        ys.append(max(y, top))
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - bottom
    if overflow > 0:
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    underflow = top - ys[0]
    if underflow > 0:
        ys = [y + underflow for y in ys]
    return {point: y for (point, _, _), y in zip(ordered, ys)}


def _svg_marker(kind: str, x: float, y: float, size: float, fill: str, stroke: str, stroke_width: float) -> str:
    marker = KIND_MARKERS.get(kind, "square")
    if marker == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    if marker == "diamond":
        pts = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    elif marker == "triangle":
        pts = [(x, y - size), (x + size, y + size), (x - size, y + size)]
    else:
        pts = [(x - size, y - size), (x + size, y - size), (x + size, y + size), (x - size, y + size)]
    points = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
    return f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def render_svg(
    points: list[Point],
    *,
    title: str,
    label_slow: float,
    label_fast: float,
    extra_label_points: set[Point] | None = None,
) -> str:
    if not points:
        raise ValueError("no timed result JSON rows found")
    width, height = 1200, 900
    left, right, top, bottom = 115, 315, 80, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_times = [p.hf_time for p in points] + [p.axon_time for p in points]
    lo, hi = _nice_log_bounds(all_times)
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)

    def sx(value: float) -> float:
        return left + (math.log10(value) - log_lo) / (log_hi - log_lo) * plot_w

    def sy(value: float) -> float:
        return top + (log_hi - math.log10(value)) / (log_hi - log_lo) * plot_h

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:12px}.axis{font-size:14px}.title{font-size:22px;font-weight:700}.legend{font-size:13px}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{left}" y="38">{escape(title)}</text>',
        f'<text class="small" x="{left}" y="60">log-log timing scatter; below diagonal means Axon is faster</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in _log_ticks(lo, hi):
        x = sx(tick)
        y = sy(tick)
        out.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        out.append(f'<text class="small" x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{escape(_fmt_time(tick))}</text>')
        out.append(f'<text class="small" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{escape(_fmt_time(tick))}</text>')

    out.append(f'<line x1="{sx(lo):.2f}" y1="{sy(lo):.2f}" x2="{sx(hi):.2f}" y2="{sy(hi):.2f}" stroke="#111827" stroke-width="1.8"/>')
    out.append(f'<text class="small" x="{sx(hi):.2f}" y="{sy(hi) - 8:.2f}" text-anchor="end">parity</text>')
    for speedup in (2, 4, 8, 16):
        # Axon speedup guide: HF/Axon = speedup, so y = x / speedup.
        x1 = max(lo, lo * speedup)
        x2 = min(hi, hi * speedup)
        if x1 >= x2:
            continue
        y1 = x1 / speedup
        y2 = x2 / speedup
        out.append(
            f'<line x1="{sx(x1):.2f}" y1="{sy(y1):.2f}" '
            f'x2="{sx(x2):.2f}" y2="{sy(y2):.2f}" '
            'stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5 5"/>'
        )
        out.append(
            f'<text class="small" x="{sx(x2):.2f}" y="{sy(y2) - 6:.2f}" '
            f'text-anchor="end" fill="#4b5563">{speedup}x faster</text>'
        )
    out.append(f'<text class="axis" x="{left + plot_w / 2:.2f}" y="{height - 35}" text-anchor="middle">HF time</text>')
    out.append(f'<text class="axis" transform="translate(32 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">Axon time</text>')

    extra_label_points = extra_label_points or set()
    outliers = _dedupe_outlier_labels([
        p for p in points
        if p.ratio >= label_slow or (1.0 / p.ratio) >= label_fast or p in extra_label_points
    ])
    outlier_set = set(outliers)
    for p in points:
        color = TASK_COLORS.get(p.task, TASK_COLORS["unknown"])
        fill = "white" if p.is_generic else color
        stroke = color
        stroke_width = 2.4 if p.is_generic else 1.0
        x = sx(p.hf_time)
        y = sy(p.axon_time)
        out.append(f"<g><title>{escape(_label(p))}: Axon/HF={p.ratio:.3f}, HF={p.hf_time:.4g}s, Axon={p.axon_time:.4g}s</title>")
        out.append(_svg_marker(p.kind, x, y, 5.2, fill, stroke, stroke_width))
        out.append("</g>")
    left_labels: list[tuple[Point, float, float]] = []
    right_labels: list[tuple[Point, float, float]] = []
    for idx, p in enumerate(sorted(outliers, key=lambda item: item.ratio, reverse=True)):
        x = sx(p.hf_time)
        y = sy(p.axon_time)
        dy = -10 if idx % 2 == 0 else 16
        if p.ratio >= label_slow:
            left_labels.append((p, x, y))
        else:
            right_labels.append((p, x, y + dy))
    label_top = top + 16
    label_bottom = top + plot_h - 10
    left_y = _spread_label_positions(left_labels, min_gap=15, top=label_top, bottom=label_bottom)
    right_y = _spread_label_positions(right_labels, min_gap=15, top=label_top, bottom=label_bottom)
    for p, x, _ in left_labels:
            label = f"{_short_checkpoint_label(p)}: {p.ratio:.3f}x"
            out.append(
                f'<text class="small" x="{x - 16:.2f}" y="{left_y[p]:.2f}" '
                f'text-anchor="end">{escape(label)}</text>'
            )
    for p, x, _ in right_labels:
        label = f"{_short_checkpoint_label(p)}: {1.0 / p.ratio:.2f}x"
        out.append(f'<text class="small" x="{x + 8:.2f}" y="{right_y[p]:.2f}">{escape(label)}</text>')

    legend_x = left + plot_w + 35
    legend_y = top + 10
    out.append(f'<text class="legend" x="{legend_x}" y="{legend_y}" font-weight="700">Task color</text>')
    y = legend_y + 22
    for task, color in TASK_COLORS.items():
        out.append(_svg_marker("circle", legend_x + 7, y - 4, 5, color, color, 1))
        out.append(f'<text class="legend" x="{legend_x + 22}" y="{y}">{escape(task)}</text>')
        y += 22
    y += 18
    out.append(f'<text class="legend" x="{legend_x}" y="{y}" font-weight="700">Model kind</text>')
    y += 22
    for kind in KIND_MARKERS:
        out.append(_svg_marker(kind, legend_x + 7, y - 4, 5, "#d1d5db", "#374151", 1.5))
        out.append(f'<text class="legend" x="{legend_x + 22}" y="{y}">{escape(kind)}</text>')
        y += 22
    y += 18
    out.append(f'<text class="legend" x="{legend_x}" y="{y}" font-weight="700">Axon file</text>')
    y += 22
    out.append(_svg_marker("circle", legend_x + 7, y - 4, 5, "white", "#2563eb", 2.4))
    out.append(f'<text class="legend" x="{legend_x + 22}" y="{y}">generic</text>')
    y += 22
    out.append(_svg_marker("circle", legend_x + 7, y - 4, 5, "#2563eb", "#2563eb", 1))
    out.append(f'<text class="legend" x="{legend_x + 22}" y="{y}">materialized/specific</text>')
    y += 35
    faster = sum(1 for p in points if p.ratio < 1)
    out.append(f'<text class="legend" x="{legend_x}" y="{y}">Rows: {len(points)}</text>')
    y += 20
    out.append(f'<text class="legend" x="{legend_x}" y="{y}">Axon faster: {faster}</text>')
    y += 20
    out.append(f'<text class="legend" x="{legend_x}" y="{y}">Axon slower/equal: {len(points) - faster}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _is_real_point(point: Point) -> bool:
    return not point.checkpoint.startswith("test/")


def _select_near_parity_real(points: list[Point], *, threshold: float, max_labels: int) -> list[Point]:
    candidates = [
        point
        for point in points
        if _is_real_point(point) and point.ratio >= threshold
    ]
    candidates.sort(key=lambda point: point.ratio, reverse=True)
    return candidates[:max_labels]


KNOWN_MODEL_LABEL_PATTERNS = (
    r"google-bert/bert-base-uncased$",
    r"openai-community/gpt2$",
    r"meta-llama/Meta-Llama-3-8B$",
    r"Qwen/Qwen3-14B$",
    r"mistralai/Mistral-7B-v0\.1$",
    r"google/gemma-4-E2B$",
    r"google-t5/t5-small$",
    r"google/gemma-7b$",
    r"Qwen/Qwen2\.5-14B$",
    r"bigscience/bloom-7b1$",
)


def _select_known_model_points(points: list[Point]) -> list[Point]:
    selected: list[Point] = []
    for pattern in KNOWN_MODEL_LABEL_PATTERNS:
        regex = re.compile(pattern)
        candidates = [point for point in points if regex.search(point.checkpoint)]
        if not candidates:
            continue
        generic = [point for point in candidates if point.is_generic]
        selected.append(min(generic or candidates, key=lambda point: point.ratio))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HF-vs-Axon timing scatter plot from axon-benchmark result JSON logs or a merged CSV.")
    parser.add_argument("input", type=Path, help="Benchmark log directory containing *.result.json files, or a merged CSV.")
    parser.add_argument("--output", "-o", type=Path, default=Path("tmp/axon-speedup-scatter.svg"))
    parser.add_argument("--title", default="HF vs Axon Runtime")
    parser.add_argument("--label-slow", type=float, default=1.25, help="Label rows with Axon/HF at or above this ratio.")
    parser.add_argument("--label-fast", type=float, default=4.0, help="Label rows where HF/Axon is at or above this speedup.")
    parser.add_argument("--label-near-parity-real", action="store_true", help="Label real non-test rows closest to/slower than the parity line.")
    parser.add_argument("--label-known-models", action="store_true", help="Label a fixed set of well-known representative checkpoints.")
    parser.add_argument("--near-parity-threshold", type=float, default=0.85, help="Minimum Axon/HF ratio for --label-near-parity-real.")
    parser.add_argument("--max-near-parity-labels", type=int, default=12, help="Maximum labels for --label-near-parity-real.")
    parser.add_argument("--normalized-128", action="store_true", help="For merged CSV input, plot *_norm128 timing columns when present.")
    args = parser.parse_args()

    points = _collect_points(args.input, normalized_128=args.normalized_128)
    extra_label_points: set[Point] = set()
    if args.label_near_parity_real:
        extra_label_points = set(
            _select_near_parity_real(
                points,
                threshold=args.near_parity_threshold,
                max_labels=args.max_near_parity_labels,
            )
        )
    if args.label_known_models:
        extra_label_points.update(_select_known_model_points(points))
    svg = render_svg(
        points,
        title=args.title,
        label_slow=args.label_slow,
        label_fast=args.label_fast,
        extra_label_points=extra_label_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    slower = sum(1 for p in points if p.ratio >= 1.0)
    print(f"wrote {args.output} ({len(points)} points, {slower} Axon/HF >= 1.0)")


if __name__ == "__main__":
    main()
