#!/usr/bin/env python3
"""Render the README charts for nanochat-on-rtx4070 from W&B run history.

Produces light- and dark-mode PNG pairs under docs/images/ so the README can
serve the right one via <picture> + prefers-color-scheme.

    python tools/reporting/export_wandb_charts.py --entity sunshines-gmail-com
    python tools/reporting/export_wandb_charts.py --offline   # no W&B, milestones only

Requires: matplotlib, and (unless --offline) wandb.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# Theme. Categorical hues are assigned in fixed order and never cycled; the
# dark column is the same hues re-stepped for the dark surface, not a flip.
# Validated with the dataviz palette validator (adjacent pairlist, both modes).
# --------------------------------------------------------------------------

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        text_primary="#0b0b0b",
        text_secondary="#52514e",
        muted="#8a8985",
        grid="#e5e4e0",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    ),
    "dark": dict(
        surface="#1a1a19",
        text_primary="#ffffff",
        text_secondary="#c3c2b7",
        muted="#8a8985",
        grid="#33322f",
        series=["#3987e5", "#d95926", "#199e70", "#c98500"],
    ),
}

GPT2_CORE = 0.256525

# --------------------------------------------------------------------------
# Ground truth from the fixed skip5120 base-eval path and the 1000-problem
# confirm packs. Used directly for the milestone charts, and as the fallback
# when W&B is unreachable.
# --------------------------------------------------------------------------

BASE_SEGMENTS = [
    # label, start step, end step, train bpb, val bpb, CORE, hours
    ("r24", 307584, 615173, 0.799226, 0.908073, 0.1494, 199.25),
    ("r32", 615173, 820230, 0.801367, 0.919863, 0.1514, 132.74),
    ("r40", 820230, 1025288, 0.784647, 0.924377, 0.1440, 132.80),
]

DEPTH_SWEEP = [
    # label, params (M), tok/s, peak VRAM (GB)
    ("d6", 73.5, 160_000, 1.1),
    ("d12", 280.0, 40_000, 3.9),
    ("d16", 537.0, 19_000, 7.1),
    ("d18", 710.0, 13_000, 9.3),
    ("d20\naspect 48", 599.0, 17_700, 7.9),
]

CONFIRM_PACK = [
    # short label, GSM8K pass@8 %, MMLU %, SpellingBee %
    ("lora_nextbest", 0.20, 21.40, 0.00),
    ("partial fr18", 0.80, 26.60, 0.39),
    ("partial fr20", 1.20, 27.40, 0.00),
    ("fr20 mixv1", 3.80, 26.60, 0.00),
    ("fr20 mixv2", 4.60, 27.40, 0.39),
    ("manual_v1 s42", 6.70, 27.60, 0.00),
]

# Label placement, tuned by eye against the rendered scatter so nothing collides.
SFT_LABEL_OFFSETS = {
    "lora_nextbest": (0, -20, "center"),
    "partial fr18 1k": (0, 12, "center"),
    "partial fr18": (-10, -20, "right"),
    "partial fr20": (0, 12, "center"),
    "fr20 mixv1": (10, -20, "left"),
    "fr20 mixv2": (0, 12, "center"),
    "manual_v1 s42": (0, 12, "center"),
}

SFT_MILESTONES = [
    # label, best val bpb, MMLU % (for the annotation)
    ("lora_nextbest", 0.6151, 21.40),
    ("partial fr18 1k", 0.5975, 23.20),
    ("partial fr18", 0.6563, 26.60),
    ("partial fr20", 0.6610, 27.40),
    ("fr20 mixv1", 0.6670, 26.60),
    ("fr20 mixv2", 0.6981, 27.40),
    ("manual_v1 s42", 0.8081, 27.60),
]


# --------------------------------------------------------------------------
# Figure scaffolding
# --------------------------------------------------------------------------


def new_fig(theme: dict, size=(9.0, 5.0), ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=size, facecolor=theme["surface"])
    axes = [axes] if ncols == 1 else list(axes)
    for ax in axes:
        ax.set_facecolor(theme["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["grid"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=theme["text_secondary"], labelsize=9, length=0)
        ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    return fig, (axes[0] if ncols == 1 else axes)


def titles(fig, theme, title, subtitle=None):
    fig.text(0.012, 0.965, title, ha="left", va="top", fontsize=13.5,
             color=theme["text_primary"], fontweight="600")
    if subtitle:
        fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=9.5,
                 color=theme["text_secondary"])


def save(fig, outdir: Path, stem: str, mode: str, theme: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}-{mode}.png"
    fig.savefig(path, dpi=180, facecolor=theme["surface"], bbox_inches="tight",
                pad_inches=0.28)
    plt.close(fig)
    print(f"  wrote {path}")


def thousands(x, _):
    return f"{x/1000:.0f}k"


# --------------------------------------------------------------------------
# W&B access
# --------------------------------------------------------------------------


def fetch_history(entity, project, name_pattern, keys, max_runs=12):
    """Return [(run_name, [(step, value), ...]), ...] for the first metric key found."""
    try:
        import wandb
    except ImportError:
        print("  wandb not installed — falling back to milestone data", file=sys.stderr)
        return []
    try:
        api = wandb.Api(timeout=30)
        runs = list(api.runs(f"{entity}/{project}"))
    except Exception as exc:  # noqa: BLE001 - any API/auth failure should degrade, not crash
        print(f"  could not reach {entity}/{project}: {exc}", file=sys.stderr)
        return []

    rx = re.compile(name_pattern) if name_pattern else None
    series = []
    for run in runs:
        if rx and not rx.search(run.name or ""):
            continue
        key = next((k for k in keys if k in run.summary or k in (run.history_keys or {}).get("keys", {})), keys[0])
        try:
            rows = [(r.get("step"), r.get(key)) for r in run.scan_history(keys=["step", key])]
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping run {run.name}: {exc}", file=sys.stderr)
            continue
        rows = [(s, v) for s, v in rows if s is not None and v is not None]
        if len(rows) >= 2:
            series.append((run.name, sorted(rows)))
        if len(series) >= max_runs:
            break
    return series


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def chart_base_val_bpb(outdir, mode, series):
    theme = THEMES[mode]
    fig, ax = new_fig(theme, (9.0, 5.2))
    titles(fig, theme,
           "Validation bits-per-byte across three base continuations",
           "d24_asp48_track, 910.7M params, 1x RTX 4070 12GB · fixed skip5120 base-eval path · lower is better")

    if series:
        for i, (name, rows) in enumerate(series[:4]):
            c = theme["series"][i % len(theme["series"])]
            xs = [s for s, _ in rows]
            ys = [v for _, v in rows]
            ax.plot(xs, ys, color=c, linewidth=2.0, label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=theme["text_secondary"], fontsize=8.5, va="center")
    else:
        for i, (label, s0, s1, _tr, val, _core, _h) in enumerate(BASE_SEGMENTS):
            c = theme["series"][i]
            ax.plot([s0, s1], [val, val], color=c, linewidth=2.0,
                    linestyle=(0, (2, 3)), solid_capstyle="round", label=label)
            ax.scatter([s1], [val], s=42, color=c, zorder=3,
                       edgecolor=theme["surface"], linewidth=2)
            ax.annotate(f"{label}  {val:.4f}", (s1, val), xytext=(8, 0),
                        textcoords="offset points", color=theme["text_primary"],
                        fontsize=9, va="center", fontweight="600")

    ax.set_xlabel("training step", color=theme["text_secondary"], fontsize=9.5)
    ax.set_ylabel("val bits-per-byte", color=theme["text_secondary"], fontsize=9.5)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlim(280_000, 1_180_000)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(theme["text_secondary"])
    fig.text(0.012, 0.015,
             "Dashes span each continuation segment; the dot is its measured base-eval value. "
             "r40 improved train bpb (0.8014 -> 0.7846) but regressed val bpb. Promoted checkpoint is r32 @ 820,230.",
             fontsize=8.5, color=theme["muted"])
    fig.subplots_adjust(top=0.80, bottom=0.16)
    save(fig, outdir, "base-val-bpb", mode, theme)


def chart_base_core(outdir, mode, series):
    theme = THEMES[mode]
    fig, ax = new_fig(theme, (9.0, 5.2))
    titles(fig, theme,
           "CORE against the GPT-2 reference",
           "The gap is the result. One consumer GPU reaches ~59% of GPT-2's CORE score.")

    if series:
        for i, (name, rows) in enumerate(series[:4]):
            c = theme["series"][i % len(theme["series"])]
            ax.plot([s for s, _ in rows], [v for _, v in rows], color=c,
                    linewidth=2.0, label=name)
    else:
        xs = [seg[2] for seg in BASE_SEGMENTS]
        ys = [seg[5] for seg in BASE_SEGMENTS]
        ax.plot(xs, ys, color=theme["series"][0], linewidth=2.0,
                marker="o", markersize=8, markeredgecolor=theme["surface"],
                markeredgewidth=2, label="d24_asp48_track")
        for (label, _s0, s1, _tr, _val, core, _h) in BASE_SEGMENTS:
            ax.annotate(f"{label}  {core:.4f}", (s1, core), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        color=theme["text_primary"], fontsize=9, fontweight="600")

    ax.axhline(GPT2_CORE, color=theme["muted"], linewidth=1.4, linestyle=(0, (5, 4)))
    ax.annotate(f"GPT-2 (1.6B) = {GPT2_CORE:.4f}", (0.985, GPT2_CORE),
                xycoords=("axes fraction", "data"), xytext=(0, 7),
                textcoords="offset points", ha="right", fontsize=9,
                color=theme["text_secondary"])

    ax.set_xlabel("training step", color=theme["text_secondary"], fontsize=9.5)
    ax.set_ylabel("CORE (DCLM)", color=theme["text_secondary"], fontsize=9.5)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(0.10, 0.30)
    if not series:
        ax.set_xlim(560_000, 1_120_000)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(theme["text_secondary"])
    fig.subplots_adjust(top=0.80, bottom=0.14)
    save(fig, outdir, "base-core", mode, theme)


def chart_sft_val_bpb(outdir, mode, series):
    theme = THEMES[mode]
    fig, ax = new_fig(theme, (9.4, 5.4))
    titles(fig, theme,
           "SFT loss and downstream capability moved in opposite directions",
           "Best SFT validation bpb on the project scored below the 25% MMLU random baseline.")

    if series:
        for i, (name, rows) in enumerate(series[:4]):
            c = theme["series"][i % len(theme["series"])]
            xs = [s for s, _ in rows]
            ys = [v for _, v in rows]
            ax.plot(xs, ys, color=c, linewidth=2.0, label=name)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        color=theme["text_secondary"], fontsize=8.5, va="center")
        ax.set_xlabel("SFT step", color=theme["text_secondary"], fontsize=9.5)
        ax.set_ylabel("val bits-per-byte", color=theme["text_secondary"], fontsize=9.5)
    else:
        labels = [m[0] for m in SFT_MILESTONES]
        bpb = [m[1] for m in SFT_MILESTONES]
        mmlu = [m[2] for m in SFT_MILESTONES]
        ax.scatter(bpb, mmlu, s=130, color=theme["series"][0], zorder=3,
                   edgecolor=theme["surface"], linewidth=2)
        for label, x, y in zip(labels, bpb, mmlu):
            dx, dy, ha = SFT_LABEL_OFFSETS.get(label, (0, 12, "center"))
            ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, fontsize=8.5, color=theme["text_primary"])
        ax.axhline(25.0, color=theme["muted"], linewidth=1.4, linestyle=(0, (5, 4)))
        ax.annotate("25% random baseline (4-choice)", (0.985, 25.0),
                    xycoords=("axes fraction", "data"), xytext=(0, -16),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=theme["text_secondary"])
        ax.set_xlabel("best SFT val bits-per-byte  (lower = better fit)",
                      color=theme["text_secondary"], fontsize=9.5)
        ax.set_ylabel("MMLU %  (higher = better capability)",
                      color=theme["text_secondary"], fontsize=9.5)
        ax.set_xlim(0.575, 0.840)
        ax.set_ylim(19, 31)

    fig.text(0.012, 0.015,
             "If loss predicted capability these points would trend down-right. They trend up-right.",
             fontsize=8.5, color=theme["muted"])
    fig.subplots_adjust(top=0.80, bottom=0.17)
    save(fig, outdir, "sft-val-bpb", mode, theme)


def chart_eval_confirm(outdir, mode):
    theme = THEMES[mode]
    fig, ax = new_fig(theme, (9.6, 5.4))
    titles(fig, theme,
           "1000-problem confirm pack across SFT candidates",
           "GSM8K pass@8 · MMLU · SpellingBee. Each bar is ~4-5 GPU-hours of eval on one 4070.")

    labels = [c[0] for c in CONFIRM_PACK]
    metrics = [
        ("GSM8K pass@8", [c[1] for c in CONFIRM_PACK], theme["series"][0]),
        ("MMLU", [c[2] for c in CONFIRM_PACK], theme["series"][1]),
        ("SpellingBee", [c[3] for c in CONFIRM_PACK], theme["series"][2]),
    ]
    n = len(metrics)
    width = 0.26
    gap = 0.012  # 2px surface gap between adjacent bars
    xs = range(len(labels))

    for i, (name, vals, color) in enumerate(metrics):
        offs = [x + (i - (n - 1) / 2) * (width + gap) for x in xs]
        bars = ax.bar(offs, vals, width=width, color=color, label=name,
                      edgecolor=theme["surface"], linewidth=1.2, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=7.6, color=theme["text_secondary"])

    ax.axhline(25.0, color=theme["muted"], linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    ax.annotate("MMLU random baseline 25%", (0.015, 25.0),
                xycoords=("axes fraction", "data"), xytext=(0, 6),
                textcoords="offset points", ha="left", fontsize=9,
                color=theme["text_secondary"])

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8.6, color=theme["text_secondary"])
    ax.set_ylabel("accuracy %", color=theme["text_secondary"], fontsize=9.5)
    ax.set_ylim(0, 34)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    for t in leg.get_texts():
        t.set_color(theme["text_secondary"])
    fig.subplots_adjust(top=0.79, bottom=0.14)
    save(fig, outdir, "eval-confirm", mode, theme)


def chart_vram_scaling(outdir, mode):
    theme = THEMES[mode]
    # Two measures on different scales get two panels, never two y-axes.
    fig, axes = new_fig(theme, (10.0, 4.6), ncols=2)
    titles(fig, theme,
           "What fits in 12GB: throughput and peak VRAM by depth",
           "500-step pretraining smoke tests on one RTX 4070 12GB.")

    labels = [d[0] for d in DEPTH_SWEEP]
    xs = range(len(labels))

    tok = [d[2] for d in DEPTH_SWEEP]
    axes[0].bar(xs, tok, width=0.6, color=theme["series"][0],
                edgecolor=theme["surface"], linewidth=1.2, zorder=3)
    for x, v in zip(xs, tok):
        axes[0].annotate(f"{v/1000:.0f}k", (x, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8.4,
                         color=theme["text_secondary"])
    axes[0].set_title("throughput (tokens/sec)", fontsize=10,
                      color=theme["text_primary"], loc="left", pad=10)
    axes[0].yaxis.set_major_formatter(FuncFormatter(thousands))

    vram = [d[3] for d in DEPTH_SWEEP]
    axes[1].bar(xs, vram, width=0.6, color=theme["series"][1],
                edgecolor=theme["surface"], linewidth=1.2, zorder=3)
    for x, v in zip(xs, vram):
        axes[1].annotate(f"{v:.1f}", (x, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8.4,
                         color=theme["text_secondary"])
    axes[1].axhline(12.0, color=theme["muted"], linewidth=1.4, linestyle=(0, (5, 4)))
    axes[1].annotate("12GB card limit", (0.98, 12.0),
                     xycoords=("axes fraction", "data"), xytext=(0, -14),
                     textcoords="offset points", ha="right", fontsize=8.6,
                     color=theme["text_secondary"])
    axes[1].set_title("peak VRAM (GB)", fontsize=10,
                      color=theme["text_primary"], loc="left", pad=10)
    axes[1].set_ylim(0, 13.4)

    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8.6, color=theme["text_secondary"])
        ax.grid(axis="x", visible=False)

    fig.text(0.012, 0.02,
             "d20 at aspect ratio 48 is narrower and deeper than d18: fewer params, faster, 1.4GB cheaper. "
             "That trade is what made the 910M d24 run fit.",
             fontsize=8.5, color=theme["muted"])
    fig.subplots_adjust(top=0.76, bottom=0.19, wspace=0.22)
    save(fig, outdir, "vram-scaling", mode, theme)


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="sunshines-gmail-com")
    p.add_argument("--base-project", default="nanochat")
    p.add_argument("--sft-project", default="nanochat-sft")
    p.add_argument("--base-run-pattern", default="d24_asp48",
                   help="regex matched against base run names")
    p.add_argument("--sft-run-pattern", default="mixv2|lora_nextbest|fr20",
                   help="regex matched against SFT run names")
    p.add_argument("--outdir", default="docs/images", type=Path)
    p.add_argument("--offline", action="store_true",
                   help="skip W&B entirely and chart the milestone tables")
    args = p.parse_args()

    if args.offline:
        base, sft = [], []
    else:
        print(f"Fetching {args.entity}/{args.base_project} ...")
        base = fetch_history(args.entity, args.base_project, args.base_run_pattern,
                             ["val/bpb"])
        print(f"Fetching {args.entity}/{args.sft_project} ...")
        sft = fetch_history(args.entity, args.sft_project, args.sft_run_pattern,
                            ["val/bpb"])

    core = fetch_history(args.entity, args.base_project, args.base_run_pattern,
                         ["core_metric"]) if (base and not args.offline) else []

    for mode in ("light", "dark"):
        print(f"[{mode}]")
        chart_base_val_bpb(args.outdir, mode, base)
        chart_base_core(args.outdir, mode, core)
        chart_sft_val_bpb(args.outdir, mode, sft)
        chart_eval_confirm(args.outdir, mode)
        chart_vram_scaling(args.outdir, mode)

    print("\nDone. Reference them from the README with a <picture> block:\n")
    print('<picture>\n'
          '  <source media="(prefers-color-scheme: dark)" srcset="docs/images/base-val-bpb-dark.png">\n'
          '  <img src="docs/images/base-val-bpb-light.png" alt="Validation bpb across base continuations">\n'
          '</picture>')


if __name__ == "__main__":
    main()
