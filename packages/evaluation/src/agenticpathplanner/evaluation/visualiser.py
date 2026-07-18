"""Visualiser — path comparison figures and markdown report generation.

Public API:
    render_comparison(instance, paths, metrics, save_path)  — side-by-side subplot grid
    render_single(instance, path, title, save_path)         — single path figure
    generate_markdown_table(results)                        — comparison table as string
    save_report(results, figures_dir, output_md)            — full case study report
"""

from __future__ import annotations

import json
import math
from pathlib import Path as FSPath
from typing import Any

import numpy as np

from agenticpathplanner.benchmarks.cec2024.metrics import PathMetrics
from agenticpathplanner.core.types import MapInstance, Path


# ---------------------------------------------------------------------------
# Path figures
# ---------------------------------------------------------------------------

def render_comparison(
    instance: MapInstance,
    paths: dict[str, Path],
    metrics: dict[str, PathMetrics],
    save_path: str | FSPath | None = None,
    cols: int = 4,
) -> None:
    """Render all planners side-by-side on the same map.

    Args:
        instance:   The map instance being solved.
        paths:      {planner_name: path} for every method being compared.
        metrics:    {planner_name: PathMetrics} — used for subplot titles.
        save_path:  PNG file path.  If None, the figure is displayed.
        cols:       Number of subplot columns (rows computed automatically).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed; cannot render.")
        return

    names = list(paths.keys())
    n     = len(names)
    rows  = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for idx, name in enumerate(names):
        ax   = axes[idx]
        path = paths.get(name, [])
        m    = metrics.get(name)
        _draw_map(ax, instance, path, name, m)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{instance.map_id}  ({instance.difficulty}, {instance.height}×{instance.width})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        FSPath(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def render_single(
    instance: MapInstance,
    path: Path,
    title: str = "",
    save_path: str | FSPath | None = None,
) -> None:
    """Render a single path on the map."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_map(ax, instance, path, title, None)
    plt.tight_layout()

    if save_path:
        FSPath(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _draw_map(ax, instance: MapInstance, path: Path, name: str, m: PathMetrics | None) -> None:
    display = np.ones((*instance.grid.shape, 3), dtype=float)
    display[instance.grid == 1] = [0.15, 0.15, 0.15]

    if path:
        for r, c in path:
            if 0 <= r < instance.height and 0 <= c < instance.width:
                display[r, c] = [0.2, 0.55, 1.0]

    ax.imshow(display, origin="upper", interpolation="nearest")

    sr, sc = instance.start
    gr, gc = instance.goal
    ax.plot(sc, sr, "gs", markersize=7, zorder=5)
    ax.plot(gc, gr, "r*", markersize=9, zorder=5)

    if path and len(path) >= 2:
        rs = [p[0] for p in path]
        cs = [p[1] for p in path]
        ax.plot(cs, rs, "b-", linewidth=1.0, alpha=0.7, zorder=4)

    # Build title with metrics
    status = ""
    if m is not None:
        if m.valid:
            status = f"PLR={m.plr:.2f} SD={m.sd:.1f}"
        else:
            status = "INVALID"
    ax.set_title(f"{name}\n{status}", fontsize=7.5, pad=3)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown_table(
    map_results: dict[str, PathMetrics],
    map_id: str,
) -> str:
    """Return a markdown metric table for one map.

    Args:
        map_results: {planner_name: PathMetrics}
        map_id:      Map identifier for the header.
    """
    lines = [
        f"### {map_id}",
        "",
        "| Method | Valid | PLR | SD | SM | Path len |",
        "|--------|:-----:|----:|---:|---:|---------:|",
    ]
    for name, m in map_results.items():
        if m is None:
            lines.append(f"| {name} | — | — | — | — | — |")
        else:
            valid = "✅" if m.valid else "❌"
            plr   = f"{m.plr:.3f}"  if m.valid else "—"
            sd    = f"{m.sd:.2f}"   if m.valid else "—"
            sm    = f"{m.sm:.2f}"   if m.valid else "—"
            plen  = f"{m.path_length:.1f}" if m.valid else "—"
            lines.append(f"| {name} | {valid} | {plr} | {sd} | {sm} | {plen} |")
    return "\n".join(lines)


def save_report(
    all_results: dict[str, dict[str, PathMetrics]],
    figures_dir: str | FSPath,
    output_md: str | FSPath,
    instance_meta: dict[str, Any] | None = None,
) -> None:
    """Write a full case study markdown report.

    Args:
        all_results:   {map_id: {planner_name: PathMetrics}}
        figures_dir:   Directory containing PNG files (relative to output_md location).
        output_md:     Path for the output markdown file.
        instance_meta: Optional {map_id: MapInstance} for difficulty/size labels.
    """
    output_md  = FSPath(output_md)
    figures_dir = FSPath(figures_dir)
    rel_fig    = figures_dir.relative_to(output_md.parent) if figures_dir.is_relative_to(output_md.parent) else figures_dir

    lines = [
        "# CEC2024 Case Study — Path Planning Method Comparison",
        "",
        "Results from `experiments/run_case_study.py`.",
        "Metrics: **PLR** = Path Length Ratio (lower ↓), **SD** = Safety Distance (higher ↑), "
        "**SM** = Smoothness (lower ↓).",
        "",
    ]

    for map_id, map_results in all_results.items():
        fig_file = rel_fig / f"{map_id}.png"
        meta = (instance_meta or {}).get(map_id)
        subtitle = f"({meta.difficulty}, {meta.height}×{meta.width})" if meta else ""

        lines += [
            f"## Map: `{map_id}` {subtitle}",
            "",
            f"![{map_id}]({fig_file})",
            "",
            generate_markdown_table(map_results, map_id),
            "",
        ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding='utf-8')
    print(f"Report saved → {output_md}")
