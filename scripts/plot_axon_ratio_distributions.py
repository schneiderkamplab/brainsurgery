#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from xml.sax.saxutils import escape

TASK_COLORS = {
    "causal_lm": "#2563eb",
    "masked_lm": "#16a34a",
    "seq2seq_lm": "#dc2626",
    "unknown": "#6b7280",
}

KIND_ORDER = {"dense": 0, "moe": 1, "ssm": 2, "other": 3}
TASK_ORDER = {"causal_lm": 0, "seq2seq_lm": 1, "masked_lm": 2, "unknown": 3}

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
class GroupKey:
    task: str
    kind: str
    source: str

    @property
    def label(self) -> str:
        return f"{self.task}\n{self.kind}\n{self.source}"


def _model_kind(axon_file: str, checkpoint: str) -> str:
    text = f"{axon_file} {checkpoint}".lower()
    if any(hint in text for hint in SSM_HINTS):
        return "ssm"
    if any(hint in text for hint in MOE_HINTS):
        return "moe"
    return "dense"


def _read_groups(path: Path, *, normalized_128: bool) -> dict[GroupKey, list[float]]:
    ratio_col = "speed_ratio_axon_over_hf_norm128" if normalized_128 else "speed_ratio_axon_over_hf"
    groups: dict[GroupKey, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                ratio = float(row.get(ratio_col) or "")
            except ValueError:
                continue
            if not math.isfinite(ratio) or ratio <= 0:
                continue
            axon = row.get("axon") or row.get("axon_file") or ""
            checkpoint = row.get("checkpoint") or row.get("checkpoint_id") or ""
            task = row.get("model_task") or "unknown"
            if task not in TASK_COLORS:
                task = "unknown"
            kind = _model_kind(axon, checkpoint)
            source = "generic" if Path(axon).name.startswith("generic-") else "materialized"
            groups[GroupKey(task=task, kind=kind, source=source)].append(ratio)
    return dict(groups)


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty values")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _nice_bounds(groups: dict[GroupKey, list[float]]) -> tuple[float, float]:
    values = [value for group in groups.values() for value in group]
    hi = max(values + [1.0])
    lo = min(values + [0.0])
    pad = max((hi - lo) * 0.08, 0.05)
    return max(0.0, lo - pad), hi + pad


def _sorted_keys(groups: dict[GroupKey, list[float]]) -> list[GroupKey]:
    return sorted(
        groups,
        key=lambda k: (
            TASK_ORDER.get(k.task, 99),
            KIND_ORDER.get(k.kind, 99),
            0 if k.source == "generic" else 1,
        ),
    )


def _axis_ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / 6
    mag = 10 ** math.floor(math.log10(raw_step))
    step = min((1, 2, 5, 10), key=lambda x: abs(raw_step - x * mag)) * mag
    start = math.ceil(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        ticks.append(value)
        value += step
    return ticks


def _render_axes(
    *,
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    title: str,
    keys: list[GroupKey],
    lo: float,
    hi: float,
) -> tuple[list[str], callable, callable]:
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(index: int) -> float:
        return left + (index + 0.5) * plot_w / len(keys)

    def sy(value: float) -> float:
        return top + (hi - value) / (hi - lo) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".axis{stroke:#374151;stroke-width:1.3}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".ref{stroke:#111827;stroke-width:1.2;stroke-dasharray:6 5}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="36" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="62" font-size="13" fill="#4b5563">Axon/HF &lt; 1 means Axon faster. Groups are task x model kind x generic/materialized.</text>',
    ]
    for tick in _axis_ticks(lo, hi):
        y = sy(tick)
        out.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        out.append(f'<text x="{left-12}" y="{y+4:.2f}" font-size="12" text-anchor="end">{tick:.2g}</text>')
    if lo <= 1.0 <= hi:
        y = sy(1.0)
        out.append(f'<line class="ref" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        out.append(f'<text x="{width-right+8}" y="{y+4:.2f}" font-size="12">HF parity</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    out.append(
        f'<text transform="translate(24 {(top + height - bottom) / 2:.2f}) rotate(-90)" '
        'font-size="14" text-anchor="middle">Axon/HF runtime ratio</text>'
    )
    for i, key in enumerate(keys):
        x = sx(i)
        lines = key.label.split("\n")
        for j, line in enumerate(lines):
            out.append(
                f'<text x="{x:.2f}" y="{height-bottom+22+j*14}" font-size="11" '
                f'text-anchor="middle">{escape(line)}</text>'
            )
    return out, sx, sy


def render_box(groups: dict[GroupKey, list[float]], *, title: str) -> str:
    if not groups:
        raise ValueError("no values to plot")
    keys = _sorted_keys(groups)
    width = max(1000, 130 * len(keys) + 180)
    height = 720
    left, right, top, bottom = 85, 50, 90, 115
    lo, hi = _nice_bounds(groups)
    out, sx, sy = _render_axes(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        title=title,
        keys=keys,
        lo=lo,
        hi=hi,
    )
    plot_w = width - left - right
    box_w = min(54, plot_w / len(keys) * 0.45)
    for i, key in enumerate(keys):
        values = sorted(groups[key])
        q1 = _percentile(values, 0.25)
        q2 = median(values)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        whisker_lo = min(v for v in values if v >= q1 - 1.5 * iqr)
        whisker_hi = max(v for v in values if v <= q3 + 1.5 * iqr)
        x = sx(i)
        color = TASK_COLORS[key.task]
        fill = color if key.source == "materialized" else "#ffffff"
        stroke_dash = "4 3" if key.kind == "moe" else ("2 3" if key.kind == "ssm" else "none")
        out.append(f'<line x1="{x:.2f}" y1="{sy(whisker_lo):.2f}" x2="{x:.2f}" y2="{sy(whisker_hi):.2f}" stroke="{color}" stroke-width="1.6"/>')
        out.append(f'<line x1="{x-box_w/3:.2f}" y1="{sy(whisker_lo):.2f}" x2="{x+box_w/3:.2f}" y2="{sy(whisker_lo):.2f}" stroke="{color}" stroke-width="1.6"/>')
        out.append(f'<line x1="{x-box_w/3:.2f}" y1="{sy(whisker_hi):.2f}" x2="{x+box_w/3:.2f}" y2="{sy(whisker_hi):.2f}" stroke="{color}" stroke-width="1.6"/>')
        out.append(
            f'<rect x="{x-box_w/2:.2f}" y="{sy(q3):.2f}" width="{box_w:.2f}" height="{max(1, sy(q1)-sy(q3)):.2f}" '
            f'fill="{fill}" fill-opacity="0.42" stroke="{color}" stroke-width="2" stroke-dasharray="{stroke_dash}"/>'
        )
        out.append(f'<line x1="{x-box_w/2:.2f}" y1="{sy(q2):.2f}" x2="{x+box_w/2:.2f}" y2="{sy(q2):.2f}" stroke="#111827" stroke-width="2"/>')
        for value in values:
            if value < whisker_lo or value > whisker_hi:
                out.append(f'<circle cx="{x:.2f}" cy="{sy(value):.2f}" r="2.5" fill="{color}" fill-opacity="0.65"/>')
        out.append(f'<text x="{x:.2f}" y="{top-12}" font-size="11" text-anchor="middle" fill="#6b7280">n={len(values)}</text>')
    out.append(_legend(width - right - 245, top))
    out.append("</svg>")
    return "\n".join(out)


def _density(values: list[float], ys: list[float]) -> list[float]:
    if len(values) == 1:
        center = values[0]
        bw = max((max(ys) - min(ys)) / 30, 0.02)
    else:
        std = (sum((v - sum(values) / len(values)) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        bw = max(1.06 * std * len(values) ** -0.2, (max(ys) - min(ys)) / 80, 0.015)
        center = None
    densities = []
    for y in ys:
        if center is not None:
            d = math.exp(-0.5 * ((y - center) / bw) ** 2)
        else:
            d = sum(math.exp(-0.5 * ((y - v) / bw) ** 2) for v in values)
        densities.append(d)
    mx = max(densities) or 1.0
    return [d / mx for d in densities]


def render_violin(groups: dict[GroupKey, list[float]], *, title: str) -> str:
    if not groups:
        raise ValueError("no values to plot")
    keys = _sorted_keys(groups)
    width = max(1000, 130 * len(keys) + 180)
    height = 720
    left, right, top, bottom = 85, 50, 90, 115
    lo, hi = _nice_bounds(groups)
    out, sx, sy = _render_axes(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        title=title,
        keys=keys,
        lo=lo,
        hi=hi,
    )
    plot_w = width - left - right
    max_w = min(48, plot_w / len(keys) * 0.42)
    grid = [lo + (hi - lo) * i / 100 for i in range(101)]
    for i, key in enumerate(keys):
        values = sorted(groups[key])
        x = sx(i)
        color = TASK_COLORS[key.task]
        fill = color if key.source == "materialized" else "#ffffff"
        stroke_dash = "4 3" if key.kind == "moe" else ("2 3" if key.kind == "ssm" else "none")
        dens = _density(values, grid)
        right_points = [(x + d * max_w, sy(y)) for y, d in zip(grid, dens)]
        left_points = [(x - d * max_w, sy(y)) for y, d in reversed(list(zip(grid, dens)))]
        points = " ".join(f"{px:.2f},{py:.2f}" for px, py in right_points + left_points)
        out.append(
            f'<polygon points="{points}" fill="{fill}" fill-opacity="0.35" stroke="{color}" '
            f'stroke-width="2" stroke-dasharray="{stroke_dash}"/>'
        )
        q2 = median(values)
        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        out.append(f'<line x1="{x-max_w*.7:.2f}" y1="{sy(q2):.2f}" x2="{x+max_w*.7:.2f}" y2="{sy(q2):.2f}" stroke="#111827" stroke-width="2"/>')
        out.append(f'<line x1="{x:.2f}" y1="{sy(q1):.2f}" x2="{x:.2f}" y2="{sy(q3):.2f}" stroke="#111827" stroke-width="2"/>')
        out.append(f'<text x="{x:.2f}" y="{top-12}" font-size="11" text-anchor="middle" fill="#6b7280">n={len(values)}</text>')
    out.append(_legend(width - right - 245, top))
    out.append("</svg>")
    return "\n".join(out)


def _legend(x: int, y: int) -> str:
    lines = [
        f'<g transform="translate({x} {y})">',
        '<rect x="0" y="0" width="235" height="150" rx="8" fill="#ffffff" stroke="#d1d5db"/>',
        '<text x="14" y="24" font-size="13" font-weight="700">Encoding</text>',
    ]
    yy = 46
    for task, color in TASK_COLORS.items():
        lines.append(f'<rect x="14" y="{yy-10}" width="12" height="12" fill="{color}" fill-opacity="0.65" stroke="{color}"/>')
        lines.append(f'<text x="34" y="{yy}" font-size="12">{escape(task)}</text>')
        yy += 19
    lines.extend(
        [
            '<rect x="128" y="36" width="16" height="12" fill="#fff" stroke="#111827" stroke-width="1.5"/>',
            '<text x="152" y="47" font-size="12">generic</text>',
            '<rect x="128" y="58" width="16" height="12" fill="#111827" fill-opacity="0.35" stroke="#111827" stroke-width="1.5"/>',
            '<text x="152" y="69" font-size="12">materialized</text>',
            '<line x1="128" y1="91" x2="146" y2="91" stroke="#111827" stroke-width="2"/>',
            '<text x="152" y="95" font-size="12">dense</text>',
            '<line x1="128" y1="113" x2="146" y2="113" stroke="#111827" stroke-width="2" stroke-dasharray="4 3"/>',
            '<text x="152" y="117" font-size="12">moe</text>',
            '<line x1="128" y1="135" x2="146" y2="135" stroke="#111827" stroke-width="2" stroke-dasharray="2 3"/>',
            '<text x="152" y="139" font-size="12">ssm</text>',
            "</g>",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Axon/HF ratio distributions grouped by task/kind/source.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--box-output", type=Path, required=True)
    parser.add_argument("--violin-output", type=Path, required=True)
    parser.add_argument("--title-prefix", default="Latest max-8B")
    parser.add_argument("--normalized-128", action="store_true")
    args = parser.parse_args()

    groups = _read_groups(args.csv_path, normalized_128=args.normalized_128)
    args.box_output.parent.mkdir(parents=True, exist_ok=True)
    args.violin_output.parent.mkdir(parents=True, exist_ok=True)
    suffix = " normalized to max-len 128" if args.normalized_128 else ""
    args.box_output.write_text(
        render_box(groups, title=f"{args.title_prefix} Axon/HF grouped box plot{suffix}"),
        encoding="utf-8",
    )
    args.violin_output.write_text(
        render_violin(groups, title=f"{args.title_prefix} Axon/HF grouped violin plot{suffix}"),
        encoding="utf-8",
    )
    print(f"wrote {args.box_output}")
    print(f"wrote {args.violin_output}")


if __name__ == "__main__":
    main()
