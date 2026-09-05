from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent / "figures"


def _panel(ax, x: float, y: float, w: float, h: float, title: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#1f2937",
            facecolor="#fffdf9",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.085),
            w,
            0.085,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.043,
        title,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
        family="DejaVu Serif",
    )


def _code_block(ax, x: float, y: float, lines: list[str], *, size: int = 9) -> None:
    step = 0.032
    for i, line in enumerate(lines):
        ax.text(
            x,
            y - i * step,
            line,
            ha="left",
            va="top",
            fontsize=size,
            family="DejaVu Sans Mono",
            color="#17202a",
        )


def _callout(ax, x: float, y: float, w: float, h: float, text: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.0,
            edgecolor="#9a6b2f",
            facecolor="#f3e3cf",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.2,
        family="DejaVu Serif",
        color="#17202a",
        linespacing=1.2,
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure() -> None:
    fig = plt.figure(figsize=(15.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f1e8")

    ax.text(
        0.5,
        0.965,
        "Dense-to-Expert-MoE Upcycling: Imperative Baseline vs Declarative BrainSurgery",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.93,
        "Same conversion semantics, but explicit checkpoint surgery and validation replace handwritten control flow.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _panel(ax, 0.035, 0.13, 0.28, 0.73, "A. Reference Python Script", "#9f2f21")
    _panel(ax, 0.36, 0.13, 0.28, 0.73, "B. BrainSurgery YAML Plan", "#155e63")
    _panel(ax, 0.685, 0.13, 0.28, 0.73, "C. Built-In Validation", "#2f5f2f")

    _code_block(
        ax,
        0.055,
        0.755,
        [
            'moe_to_dense_mapping = {',
            '  "feed_forward_moe.experts.mlp.w1": ...',
            '  "feed_forward_moe.experts.mlp.w2": ...',
            '  "attention.w_q.weight": ...',
            "}",
            "",
            "for expert, path in enumerate(dense_paths):",
            "  dense_state_dict = load_state_dict(path)",
            "  for key in list(moe_state_dict.keys()):",
            "    if any(pattern in key for pattern in ...):",
            "      dense_key = key.replace(...)",
            '      if "expert" in key or "router" in key:',
            "        moe_state_dict[key][...] =",
            "          dense_state_dict[dense_key].T",
            "      ...",
        ],
    )
    _callout(
        ax,
        0.05,
        0.145,
        0.25,
        0.07,
        "Control flow, mapping, mutation,\nand output writing are intertwined.",
    )

    _code_block(
        ax,
        0.38,
        0.755,
        [
            "transforms:",
            "  - assert: { exists: m0::model.embed_tokens.weight }",
            "  - assert:",
            "      equal:",
            r"        left:  m0::model.layers\.(\d+)\.mlp...",
            r"        right: m1::model.layers.\1.mlp...",
            "  - copy:",
            r"      from: m0::model.layers\.(\d+)\.mlp...",
            r"      to:   m0::model.layers.\1.mlp.experts.0...",
            "  - copy:",
            r"      from: m1::model.layers\.(\d+)\.mlp...",
            r"      to:   m0::model.layers.\1.mlp.experts.1...",
            "  - fill:",
            r"      to: m0::model.layers.\1.mlp.gate.weight",
            "      mode: constant",
            "      value: 0",
            r"  - delete: { target: m0::model.layers\.(\d+)\.mlp... }",
        ],
    )
    _callout(
        ax,
        0.375,
        0.145,
        0.25,
        0.07,
        "Assertions, surgery, and validation are\nexplicit, reviewable, and reusable.",
    )

    _code_block(
        ax,
        0.705,
        0.755,
        [
            "inputs:",
            "  - yaml::olmo_1b_0724_hf_dense_moe_demo",
            "  - ref::olmo_1b_0724_hf_dense_moe_reference",
            "",
            "transforms:",
            "  - diff: { mode: aliases, left_alias: ref,",
            "            right_alias: yaml }",
            "",
            "Missing on left:",
            "  (none)",
            "Missing on right:",
            "  (none)",
            "Differing:",
            "  (none)",
            "No differences found.",
        ],
    )
    _callout(
        ax,
        0.7,
        0.145,
        0.25,
        0.07,
        "The declarative plan matches the\nindependent reference implementation.",
    )

    ax.add_patch(
        FancyArrowPatch((0.315, 0.495), (0.36, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )
    ax.add_patch(
        FancyArrowPatch((0.64, 0.495), (0.685, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )

    ax.text(
        0.5,
        0.07,
        "Dense checkpoint A + Dense checkpoint B  ->  conversion  ->  MoE-style checkpoint  ->  BrainSurgery diff",
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Serif",
        color="#17202a",
    )

    _save(fig, "olmo_1b_0724_comparison")


def make_pipeline_figure() -> None:
    fig = plt.figure(figsize=(13.5, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.93,
        "Validated BrainSurgery Workflow for Dense-to-Expert-MoE Upcycling",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )

    boxes = [
        (0.04, 0.42, 0.17, 0.2, "#e8eef9", "Dense\ncheckpoint A"),
        (0.04, 0.14, 0.17, 0.2, "#e8eef9", "Dense\ncheckpoint B"),
        (0.29, 0.28, 0.19, 0.2, "#dff3f1", "BrainSurgery\nYAML plan"),
        (0.55, 0.42, 0.17, 0.2, "#eef7e8", "Converted\nMoE checkpoint"),
        (0.55, 0.14, 0.17, 0.2, "#f7efe8", "Reference\nPython output"),
        (0.80, 0.28, 0.16, 0.2, "#edf7ed", "Diff result:\nNo differences\nfound"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.15,
        )

    arrows = [
        ((0.21, 0.52), (0.29, 0.40)),
        ((0.21, 0.24), (0.29, 0.36)),
        ((0.48, 0.39), (0.55, 0.52)),
        ((0.21, 0.24), (0.55, 0.24)),
        ((0.72, 0.52), (0.80, 0.39)),
        ((0.72, 0.24), (0.80, 0.37)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    ax.text(
        0.5,
        0.06,
        "BrainSurgery externalizes checkpoint surgery as a declarative, executable,\nand verifiable artifact.",
        ha="center",
        va="center",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
        linespacing=1.2,
    )

    _save(fig, "olmo_1b_0724_pipeline")


def make_intro_workflow_figure() -> None:
    fig = plt.figure(figsize=(15.0, 6.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf7")

    def _stage_label(x: float, text: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.06, 0.86),
                0.12,
                0.042,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(
            x,
            0.881,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            family="DejaVu Sans",
            color="white",
        )

    def _pill(x: float, y: float, w: float, text: str, *, fc: str, ec: str, tc: str = "#17202a") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                0.045,
                boxstyle="round,pad=0.008,rounding_size=0.022",
                linewidth=1.0,
                edgecolor=ec,
                facecolor=fc,
            )
        )
        ax.text(
            x + w / 2,
            y + 0.0225,
            text,
            ha="center",
            va="center",
            fontsize=9.2,
            family="DejaVu Sans",
            color=tc,
            fontweight="bold",
        )

    def _brain_cluster(
        x: float,
        y: float,
        *,
        face: str,
        edge: str,
        title: str,
        subtitle: str,
        node_color: str,
        accent: str,
    ) -> None:
        for center, width, height in [
            ((x - 0.035, y + 0.035), 0.13, 0.17),
            ((x + 0.04, y + 0.04), 0.14, 0.18),
            ((x + 0.0, y - 0.015), 0.19, 0.16),
        ]:
            ax.add_patch(
                Ellipse(
                    center,
                    width,
                    height,
                    linewidth=1.8,
                    edgecolor=edge,
                    facecolor=face,
                    zorder=1,
                )
            )

        ax.plot([x - 0.005, x - 0.005], [y - 0.08, y + 0.13], color=edge, lw=1.2, alpha=0.65, zorder=2)
        ax.plot([x - 0.08, x + 0.08], [y + 0.015, y + 0.015], color=edge, lw=1.0, alpha=0.35, zorder=2)
        ax.plot([x - 0.075, x + 0.075], [y + 0.08, y - 0.03], color=edge, lw=1.0, alpha=0.35, zorder=2)
        ax.plot([x - 0.07, x + 0.07], [y - 0.02, y + 0.09], color=edge, lw=1.0, alpha=0.35, zorder=2)

        nodes = [
            (x - 0.065, y + 0.07),
            (x - 0.03, y + 0.11),
            (x + 0.025, y + 0.085),
            (x + 0.07, y + 0.045),
            (x - 0.055, y - 0.005),
            (x, y + 0.015),
            (x + 0.055, y - 0.005),
        ]
        for nx, ny in nodes:
            ax.add_patch(Circle((nx, ny), 0.009, facecolor=node_color, edgecolor="white", linewidth=0.8, zorder=3))

        ax.text(
            x,
            y + 0.185,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            family="DejaVu Serif",
            color="#17202a",
        )
        ax.text(
            x,
            y - 0.135,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            family="DejaVu Serif",
            color="#475569",
            linespacing=1.2,
        )
        _pill(x - 0.07, y - 0.21, 0.14, "checkpoint artifact", fc="#fff7ed", ec=accent)

    def _mini_card(x: float, y: float, w: float, h: float, title: str, lines: list[str], *, accent: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.016",
                linewidth=1.2,
                edgecolor="#d0d7e2",
                facecolor="white",
            )
        )
        ax.add_patch(Rectangle((x, y + h - 0.042), w, 0.042, linewidth=0, facecolor=accent))
        ax.text(
            x + 0.015,
            y + h - 0.021,
            title,
            ha="left",
            va="center",
            fontsize=10,
            family="DejaVu Sans",
            color="white",
            fontweight="bold",
        )
        _code_block(ax, x + 0.015, y + h - 0.055, lines, size=8.1)

    ax.text(
        0.5,
        0.952,
        "BrainSurgery as Surgical Checkpoint Editing",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.912,
        "Declarative plans, interactive inspection, and executable validation turn checkpoint rewrites into explicit research artifacts.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _stage_label(0.17, "Source", "#516b91")
    _stage_label(0.50, "Operate", "#b8562b")
    _stage_label(0.83, "Validate", "#5a7b52")

    _brain_cluster(
        0.17,
        0.53,
        face="#fde9d8",
        edge="#c36d35",
        title="Artificial brain before surgery",
        subtitle="Dense checkpoint, tensor families,\nand implicit script logic.",
        node_color="#7c3f00",
        accent="#d7a46a",
    )
    _pill(0.08, 0.20, 0.11, "input A", fc="#eef4fb", ec="#9cb5d9")
    _pill(0.205, 0.20, 0.11, "input B", fc="#eef4fb", ec="#9cb5d9")

    ax.add_patch(
        FancyArrowPatch((0.28, 0.53), (0.39, 0.53), connectionstyle="arc3,rad=0.0", arrowstyle="-|>", mutation_scale=22, lw=2.6, color="#b8562b")
    )
    ax.add_patch(
        FancyArrowPatch((0.61, 0.53), (0.72, 0.53), connectionstyle="arc3,rad=0.0", arrowstyle="-|>", mutation_scale=22, lw=2.6, color="#5a7b52")
    )

    ax.add_patch(Circle((0.50, 0.53), 0.13, facecolor="#fff4ec", edgecolor="#dfb393", linewidth=1.5))
    ax.text(
        0.50,
        0.665,
        "Surgical intervention",
        ha="center",
        va="bottom",
        fontsize=14,
        family="DejaVu Serif",
        color="#17202a",
        fontweight="bold",
    )
    _mini_card(
        0.43,
        0.50,
        0.14,
        0.12,
        "Plan",
        ["copy", "slice", "assert", "phlora"],
        accent="#155e63",
    )
    ax.plot([0.468, 0.535], [0.44, 0.595], color="#6b7280", lw=3.0, solid_capstyle="round")
    ax.add_patch(Rectangle((0.533, 0.59), 0.028, 0.011, angle=-26, facecolor="#d45d1f", edgecolor="#9a3f10", linewidth=0.8))
    ax.add_patch(Circle((0.50, 0.46), 0.008, facecolor="#d45d1f", edgecolor="white", linewidth=0.5))
    ax.add_patch(Circle((0.525, 0.48), 0.008, facecolor="#d45d1f", edgecolor="white", linewidth=0.5))
    ax.add_patch(Circle((0.548, 0.505), 0.008, facecolor="#d45d1f", edgecolor="white", linewidth=0.5))
    _pill(0.415, 0.37, 0.10, "assert", fc="#ecfdf3", ec="#76b38c")
    _pill(0.525, 0.37, 0.09, "diff", fc="#eef8ee", ec="#83a97c")
    _callout(
        ax,
        0.39,
        0.25,
        0.22,
        0.08,
        "Plans make the intervention explicit;\nvalidation makes it inspectable.",
    )

    _brain_cluster(
        0.83,
        0.53,
        face="#eaf5ea",
        edge="#5a7b52",
        title="Artificial brain after surgery",
        subtitle="Rewritten checkpoint, executable\ninvariants, and validated outputs.",
        node_color="#2f6b5f",
        accent="#8bb48c",
    )
    _pill(0.76, 0.20, 0.14, "rewritten artifact", fc="#eef8ee", ec="#8bb48c")

    _mini_card(
        0.07,
        0.07,
        0.23,
        0.10,
        "Interfaces",
        ["batch CLI", "interactive CLI", "WebUI"],
        accent="#516b91",
    )
    _mini_card(
        0.36,
        0.07,
        0.28,
        0.10,
        "WebUI signals",
        ["preview impact", "checkpoint diff", "results + current models"],
        accent="#155e63",
    )
    _mini_card(
        0.70,
        0.07,
        0.24,
        0.10,
        "Reproducible summary",
        ["executed plan", "resolved transforms", "shareable validation trail"],
        accent="#5a7b52",
    )

    ax.text(
        0.5,
        0.015,
        "BrainSurgery turns checkpoint editing into a precise workflow: plan the intervention, inspect its effect, validate it, and keep the executed summary as a reproducible artifact.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        family="DejaVu Serif",
        color="#334155",
    )

    _save(fig, "brainsurgery_intro_workflow")


def make_low_rank_figure() -> None:
    fig = plt.figure(figsize=(14.4, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbf8f3")

    ax.text(
        0.5,
        0.945,
        "Low-Rank and PHLoRA Expert Rewrites as Declarative Checkpoint Surgery",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.902,
        "A validated MoE checkpoint becomes the common starting point for two complementary expert-compression workflows.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    boxes = [
        (0.05, 0.37, 0.18, 0.22, "#e8eef9", "Validated\n2-expert MoE\ncheckpoint"),
        (0.31, 0.52, 0.22, 0.20, "#dff3f1", "BrainSurgery YAML:\nPHLoRA factorization\nof expert-1 deltas"),
        (0.31, 0.16, 0.22, 0.20, "#dff3f1", "BrainSurgery YAML:\nlow-rank in-place\nexpert rewrite"),
        (0.61, 0.52, 0.16, 0.20, "#eef7e8", "FlexMoRE-style\nPHLoRA output"),
        (0.61, 0.16, 0.16, 0.20, "#eef7e8", "Dense MoE with\nlow-rank expert 1"),
        (0.82, 0.52, 0.13, 0.20, "#edf7ed", "Reference\nPython\n+ diff"),
        (0.82, 0.16, 0.13, 0.20, "#edf7ed", "Reference\nPython\n+ diff"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.15,
        )

    arrows = [
        ((0.23, 0.52), (0.31, 0.62)),
        ((0.23, 0.45), (0.31, 0.26)),
        ((0.53, 0.62), (0.61, 0.62)),
        ((0.53, 0.26), (0.61, 0.26)),
        ((0.77, 0.62), (0.82, 0.62)),
        ((0.77, 0.26), (0.82, 0.26)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    _callout(
        ax,
        0.305,
        0.75,
        0.235,
        0.075,
        "Expert 1 becomes explicit PHLoRA factors\nrelative to dense expert 0.",
    )
    _callout(
        ax,
        0.305,
        0.055,
        0.235,
        0.075,
        "Expert 1 stays a standard dense tensor\nbut is rewritten by a rank-limited delta.",
    )

    ax.text(
        0.5,
        0.025,
        "Both branches stay in the same reproducible pattern: YAML surgery -> reference implementation -> diff-based validation.",
        ha="center",
        va="center",
        fontsize=10.5,
        family="DejaVu Serif",
        color="#334155",
    )

    _save(fig, "olmo_1b_0724_low_rank")


def make_axon_synapse_figure() -> None:
    fig = plt.figure(figsize=(14.2, 5.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.94,
        "How BrainSurgery Fits with Axon and Synapse",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.895,
        "BrainSurgery edits checkpoint weights; Axon and Synapse describe executable model structure.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    boxes = [
        (0.05, 0.52, 0.2, 0.18, "#f7efe8", "Axon DSL\nhuman-authored\nmodel graph"),
        (0.36, 0.52, 0.2, 0.18, "#e8eef9", "Synapse YAML\nstructured declarative\nmodel spec"),
        (0.67, 0.52, 0.2, 0.18, "#eef7e8", "Generated / runtime\nPyTorch model"),
        (0.05, 0.17, 0.2, 0.18, "#dff3f1", "BrainSurgery YAML\ncheckpoint surgery\nplans"),
        (0.36, 0.17, 0.2, 0.18, "#edf7ed", "Converted / validated\ncheckpoint artifacts"),
        (0.67, 0.17, 0.2, 0.18, "#f3e3cf", "Bridge example:\nAxon graph aligned to\nrewritten checkpoint"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.18,
        )

    arrows = [
        ((0.25, 0.61), (0.36, 0.61)),
        ((0.56, 0.61), (0.67, 0.61)),
        ((0.25, 0.26), (0.36, 0.26)),
        ((0.56, 0.26), (0.67, 0.26)),
        ((0.46, 0.35), (0.46, 0.52)),
        ((0.77, 0.35), (0.77, 0.52)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    _callout(
        ax,
        0.31,
        0.77,
        0.30,
        0.07,
        "Axon is the readable authoring language; Synapse is the structured model specification.",
    )
    _callout(
        ax,
        0.31,
        0.04,
        0.30,
        0.07,
        "Checkpoint surgery and executable model structure stay separate, but connect cleanly.",
    )

    _save(fig, "olmo_1b_0724_axon_synapse")


def make_low_rank_comparison_figure() -> None:
    fig = plt.figure(figsize=(15.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f1e8")

    ax.text(
        0.5,
        0.965,
        "Low-Rank Expert Rewriting: Imperative Reference vs Declarative BrainSurgery",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.93,
        "The same declarative pattern extends from structural MoE upcycling to representation-level expert compression.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _panel(ax, 0.035, 0.13, 0.28, 0.73, "A. Reference Python Script", "#9f2f21")
    _panel(ax, 0.36, 0.13, 0.28, 0.73, "B. BrainSurgery YAML Plan", "#155e63")
    _panel(ax, 0.685, 0.13, 0.28, 0.73, "C. Built-In Validation", "#2f5f2f")

    _code_block(
        ax,
        0.055,
        0.755,
        [
            "for layer in range(16):",
            '  for proj in ("gate_proj", "up_proj", "down_proj"):',
            '    expert0_key = f"model.layers.{layer}.mlp..."',
            '    expert1_key = f"model.layers.{layer}.mlp..."',
            "    delta = source[expert1_key] - source[expert0_key]",
            "    approx_delta = reconstruct_phlora_rank(",
            "      delta,",
            "      rank,",
            "      cache=cache,",
            "      cache_key=expert1_key,",
            "      ...",
            "    )",
            "    out[expert1_key] =",
            "      source[expert0_key] + approx_delta",
        ],
    )
    _callout(
        ax,
        0.05,
        0.145,
        0.25,
        0.07,
        "Low-rank approximation logic is handwritten\ninside loops and tensor mutation code.",
    )

    _code_block(
        ax,
        0.38,
        0.755,
        [
            "transforms:",
            "  - copy: { from: expert_1, to: expert_1.delta }",
            "  - subtract_:",
            "      from: model::...experts.0...",
            "      to:   model::...experts.1...delta...",
            "  - phlora:",
            "      target:   model::...experts.1...delta...",
            "      target_a: model::...phlora_a.weight",
            "      target_b: model::...phlora_b.weight",
            "      rank: 64",
            "  - delete: { target: model::...experts.1.weight }",
            "  - assert:",
            "      shape: { of: model::...phlora_a.weight, is: [64, ...] }",
            "  - assert:",
            "      not: { exists: model::...experts.1.weight }",
        ],
    )
    _callout(
        ax,
        0.375,
        0.145,
        0.25,
        0.07,
        "Compression, cleanup, and safety checks are\nexpressed directly in the checkpoint plan.",
    )

    _code_block(
        ax,
        0.705,
        0.755,
        [
            "inputs:",
            "  - yaml::olmo_1b_0724_hf_low_rank_expert_r64_demo",
            "  - ref::olmo_1b_0724_hf_low_rank_expert_r64_reference",
            "",
            "transforms:",
            "  - diff: { mode: aliases, left_alias: ref,",
            "            right_alias: yaml }",
            "",
            "Missing on left:",
            "  (none)",
            "Missing on right:",
            "  (none)",
            "Differing:",
            "  (none)",
            "No differences found.",
        ],
    )
    _callout(
        ax,
        0.7,
        0.145,
        0.25,
        0.07,
        "The low-rank rewrite is also validated\nagainst an independent reference.",
    )

    ax.add_patch(
        FancyArrowPatch((0.315, 0.495), (0.36, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )
    ax.add_patch(
        FancyArrowPatch((0.64, 0.495), (0.685, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )

    ax.text(
        0.5,
        0.07,
        "Validated MoE checkpoint  ->  low-rank expert rewrite  ->  reference implementation  ->  BrainSurgery diff",
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Serif",
        color="#17202a",
    )

    _save(fig, "olmo_1b_0724_low_rank_comparison")


def make_targeting_validation_figure() -> None:
    fig = plt.figure(figsize=(15.2, 8.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f1e8")

    ax.text(
        0.5,
        0.965,
        "Core BrainSurgery Primitives: Bulk Targeting, Slicing, and Assertion-Backed Validation",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.928,
        "Before the MoE and PHLoRA case studies, the reader sees the smaller ideas that make those larger workflows possible.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _panel(ax, 0.035, 0.13, 0.28, 0.73, "A. Bulk Tensor Targeting", "#155e63")
    _panel(ax, 0.36, 0.13, 0.28, 0.73, "B. Targeting with Slices", "#2f5f2f")
    _panel(ax, 0.685, 0.13, 0.28, 0.73, "C. Verification as Code", "#9f2f21")

    _code_block(
        ax,
        0.055,
        0.76,
        [
            "transforms:",
            "  - cast:",
            r'      from: "model\.layers\.(\d+)\.self_attn\.',
            r'             (q_proj|k_proj|v_proj|o_proj)\.weight"',
            r'      to:   "attn_fp16.\g<0>"',
            "      dtype: float16",
            "",
            "# 16 layers x 4 projections = 64 tensors",
            "# one declarative operation, no Python loop",
        ],
    )
    _callout(
        ax,
        0.05,
        0.145,
        0.25,
        0.08,
        "Regex-based tensor references let one plan step\napply to an entire tensor family at model scale.",
    )

    _code_block(
        ax,
        0.38,
        0.76,
        [
            "transforms:",
            "  - copy:",
            r'      from: "model.layers.0.self_attn.',
            r'             q_proj.weight::[:128, :128]"',
            '      to:   "debug.layer0.q_proj_block"',
            "",
            "  - dump:",
            '      target: "debug.layer0.q_proj_block"',
            "      format: compact",
        ],
    )
    _callout(
        ax,
        0.375,
        0.145,
        0.25,
        0.08,
        "The same reference syntax supports localized edits,\ninspection, and debugging on tensor subregions.",
    )

    _code_block(
        ax,
        0.705,
        0.76,
        [
            "inputs:",
            "  - src::olmo_1b_0724_hf_dense",
            "  - out::olmo_1b_0724_hf_dense_moe_demo",
            "",
            "transforms:",
            "  - assert: { exists: out::model.layers.0.mlp.",
            "              experts.0.gate_proj.weight }",
            "  - assert:",
            "      shape: { of: out::model.layers.0.mlp.",
            "               gate.weight, is: [2, 2048] }",
            "  - assert:",
            "      equal:",
            "        left:  src::model.layers.0.mlp.",
            "               gate_proj.weight::[:16, :16]",
            "        right: out::model.layers.0.mlp.experts.0.",
            "               gate_proj.weight::[:16, :16]",
            "  - assert: { not: { exists: out::model.layers.0.",
            "              mlp.gate_proj.weight } }",
        ],
        size=8.6,
    )
    _callout(
        ax,
        0.70,
        0.145,
        0.25,
        0.08,
        "Expected postconditions become executable invariants,\nreducing silent checkpoint-editing failures.",
    )

    ax.add_patch(
        FancyArrowPatch((0.315, 0.495), (0.36, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )
    ax.add_patch(
        FancyArrowPatch((0.64, 0.495), (0.685, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )

    ax.text(
        0.5,
        0.07,
        "Model-scale selection  ->  precise sub-tensor editing  ->  executable validation",
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Serif",
        color="#17202a",
    )

    _save(fig, "brainsurgery_targeting_validation")


def main() -> None:
    make_intro_workflow_figure()
    make_targeting_validation_figure()
    make_comparison_figure()
    make_pipeline_figure()
    make_low_rank_figure()
    make_axon_synapse_figure()
    make_low_rank_comparison_figure()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
