#!/usr/bin/env python3
"""
collbench_viz.py — unified visualizer for CollBench allgather JSON exports.

Three modes
  grid       (default) — N small broadcast-trees in a grid, one per source rank
  overlay              — all N broadcast-trees stacked vertically on one canvas
  butterfly            — rank-vs-step matrix; arrows show which peers exchange
                         at each step (send left, recv right of each column)

Auto-sizing
  Figure dimensions, DPI, font sizes, node radii, and spacing are derived
  automatically from n_ranks and n_steps unless overridden on the CLI.

Usage:
    python collbench_viz.py [OPTIONS] <json_file>

Examples:
    python collbench_viz.py out/bine_allgather_b2b.json
    python collbench_viz.py --mode overlay out.json
    python collbench_viz.py --mode butterfly --no-latency out.json
    python collbench_viz.py --mode butterfly --dpi 300 --out fig.png out.json
    python collbench_viz.py --mode grid --grid-cols 4 --dpi 200 out.json
"""

import argparse
import json
import math
from collections import defaultdict

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ── fonts ─────────────────────────────────────────────────────────────────────

def _best_sans():
    available = {f.name for f in fm.fontManager.ttflist}
    for c in ["DejaVu Sans", "Liberation Sans", "FreeSans", "Noto Sans", "Ubuntu"]:
        if c in available:
            return c
    return "sans-serif"


matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_best_sans()],
    "axes.spines.left":   False,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.spines.bottom": False,
})

DEFAULT_PALETTE = [
    "#E05C5C", "#5B8FF9", "#5AD8A6", "#F6BD16", "#9370DB",
    "#FF9F7F", "#4BC9D8", "#A0D911", "#FF85C2", "#B37FEB",
]


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_ops(path):
    with open(path) as f:
        return json.load(f)["operations"]


def summarize(ops):
    ranks = sorted({o["rank"] for o in ops})
    steps = sorted({o["algo_idx"] for o in ops})
    return ranks, steps


def build_exchanges(ops):
    """Aggregate per-(rank_a, rank_b, step) pair with averaged latencies."""
    table = {}
    for o in ops:
        r, p, s = o["rank"], o["peer"], o["algo_idx"]
        key = (min(r, p), max(r, p), s)
        if key not in table:
            table[key] = dict(rank_a=min(r, p), rank_b=max(r, p), step=s,
                              send_lats=[], recv_lats=[],
                              t_start_ns=o["t_start_ns"])
        entry = table[key]
        if o["operation_type"] == "send":
            entry["send_lats"].append(o["t_total_ns"] / 1_000.0)
        else:
            entry["recv_lats"].append(o["t_total_ns"] / 1_000.0)
        entry["t_start_ns"] = min(entry["t_start_ns"], o["t_start_ns"])

    result = []
    for ex in table.values():
        ex["send_lat"] = (sum(ex["send_lats"]) / len(ex["send_lats"])
                          if ex["send_lats"] else None)
        ex["recv_lat"] = (sum(ex["recv_lats"]) / len(ex["recv_lats"])
                          if ex["recv_lats"] else None)
        result.append(ex)
    return sorted(result, key=lambda x: (x["step"], x["rank_a"]))


# ── auto-scale helpers ────────────────────────────────────────────────────────

def _auto_dpi(fig_w, fig_h, target_px=2400):
    """Choose DPI so the longer side is ~target_px pixels."""
    longer = max(fig_w, fig_h)
    return max(100, min(400, round(target_px / longer / 10) * 10))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def auto_grid_params(n_ranks, n_steps, grid_cols=None):
    """Return (ncols, nrows, cell_w, cell_h, col_w, row_h, node_r, fontsizes)."""
    ncols  = grid_cols or math.ceil(math.sqrt(n_ranks))
    nrows  = math.ceil(n_ranks / ncols)
    # each small tree has n_ranks nodes wide × n_steps+1 rows tall
    # target cell size: enough for the tree to breathe
    col_w  = _clamp(1.4 - 0.04 * n_ranks, 0.55, 1.3)
    row_h  = _clamp(1.3 - 0.04 * n_steps,  0.55, 1.2)
    cell_w = n_ranks * col_w + 0.9
    cell_h = (n_steps + 1) * row_h + 1.0
    node_r = _clamp(0.28 - 0.012 * n_ranks, 0.10, 0.26)
    node_fs  = _clamp(9 - 0.4 * n_ranks, 5, 9)
    lat_fs   = _clamp(7 - 0.3 * n_ranks, 4, 7)
    title_fs = 13
    cell_title_fs = _clamp(9 - 0.2 * n_ranks, 6, 9)
    fig_w = ncols * cell_w + 1.5
    fig_h = nrows * cell_h + 1.2
    return dict(ncols=ncols, nrows=nrows, cell_w=cell_w, cell_h=cell_h,
                col_w=col_w, row_h=row_h, node_r=node_r,
                node_fs=node_fs, lat_fs=lat_fs, title_fs=title_fs,
                cell_title_fs=cell_title_fs, fig_w=fig_w, fig_h=fig_h)


def auto_overlay_params(n_ranks, n_steps):
    col_w  = _clamp(1.4 - 0.04 * n_ranks, 0.55, 1.3)
    row_h  = _clamp(1.2 - 0.04 * n_steps,  0.50, 1.1)
    node_r = _clamp(0.26 - 0.010 * n_ranks, 0.09, 0.24)
    node_fs  = _clamp(9 - 0.4 * n_ranks, 5, 9)
    lat_fs   = _clamp(7 - 0.3 * n_ranks, 4, 7)
    fig_w  = n_ranks * col_w + 2.5
    fig_h  = n_ranks * (n_steps + 2.2) * row_h + 1.2
    return dict(col_w=col_w, row_h=row_h, node_r=node_r,
                node_fs=node_fs, lat_fs=lat_fs,
                fig_w=fig_w, fig_h=fig_h)


def auto_butterfly_params(n_ranks, n_steps):
    col_w  = _clamp(3.2 - 0.06 * n_steps,  1.5, 3.2)
    row_h  = _clamp(1.8 - 0.05 * n_ranks,  0.8, 1.8)
    node_r = _clamp(0.30 - 0.010 * n_ranks, 0.10, 0.28)
    node_fs  = _clamp(9 - 0.35 * n_ranks, 5, 9)
    lat_fs   = _clamp(7 - 0.25 * n_ranks, 4, 7)
    rank_fs  = _clamp(9 - 0.20 * n_ranks, 6, 9)
    step_fs  = _clamp(9 - 0.20 * n_steps, 6, 9)
    fig_w  = n_steps * col_w + 3.0
    fig_h  = n_ranks * row_h + 1.5
    return dict(col_w=col_w, row_h=row_h, node_r=node_r,
                node_fs=node_fs, lat_fs=lat_fs,
                rank_fs=rank_fs, step_fs=step_fs,
                fig_w=fig_w, fig_h=fig_h)


# ── broadcast-tree helpers ────────────────────────────────────────────────────

def build_broadcast_tree(source, exchanges, ranks):
    have = {source}
    node_step = {source: -1}
    edges = []
    by_step = defaultdict(list)
    for ex in exchanges:
        by_step[ex["step"]].append(ex)
    for step in sorted(by_step):
        new_have = set(have)
        for ex in by_step[step]:
            ra, rb = ex["rank_a"], ex["rank_b"]
            if ra in have and rb not in have:
                new_have.add(rb)
                node_step[rb] = step
                lat = ex["send_lat"] or ex["recv_lat"]
                edges.append((ra, rb, step, lat))
            elif rb in have and ra not in have:
                new_have.add(ra)
                node_step[ra] = step
                lat = ex["send_lat"] or ex["recv_lat"]
                edges.append((rb, ra, step, lat))
        have = new_have
    return node_step, edges


def tree_layout(source, edges, ranks):
    children = defaultdict(list)
    for parent, child, step, lat in edges:
        children[parent].append((child, step))

    col_counter = [0]
    node_col = {}

    def fill(node):
        kids = children.get(node, [])
        if not kids:
            node_col[node] = col_counter[0]
            col_counter[0] += 1
            return
        for child, _ in kids:
            fill(child)
        cols = [node_col[c] for c, _ in kids]
        node_col[node] = (cols[0] + cols[-1]) / 2.0

    fill(source)
    for r in ranks:
        if r not in node_col:
            node_col[r] = col_counter[0]
            col_counter[0] += 1

    node_step_map = {source: -1}
    for parent, child, step, _ in edges:
        node_step_map[child] = step
    node_row = {r: (0 if node_step_map.get(r, -1) == -1
                    else node_step_map[r] + 1) for r in ranks}
    return node_col, node_row


# ── shared arrow primitive ────────────────────────────────────────────────────

def _draw_arrow(ax, x0, y0, x1, y1, color, lw, node_r, mutation_scale=10):
    d = math.hypot(x1 - x0, y1 - y0)
    if d < 2 * node_r + 1e-9:
        return
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    ax.annotate("",
        xy=(x1 - ux * node_r, y1 - uy * node_r),
        xytext=(x0 + ux * node_r, y0 + uy * node_r),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=mutation_scale,
                        shrinkA=0, shrinkB=0),
        zorder=3)


# ── single-tree draw ──────────────────────────────────────────────────────────

def draw_tree(ax, source, exchanges, ranks, steps,
              ox, oy, col_w, row_h, node_r,
              color, root_color, node_color, node_edge, bg,
              show_latency, lat_fs, node_fs, ghost_color):
    node_step, edges = build_broadcast_tree(source, exchanges, ranks)
    node_col, node_row = tree_layout(source, edges, ranks)

    def pos(r):
        return ox + node_col[r] * col_w, oy - node_row[r] * row_h

    # ghost skeleton
    for parent, child, step, _ in edges:
        x0, y0 = pos(parent); x1, y1 = pos(child)
        ax.plot([x0, x1], [y0, y1], color=ghost_color, lw=0.7,
                linestyle=(0, (4, 5)), zorder=1)

    # arrows + latency
    for parent, child, step, lat in edges:
        x0, y0 = pos(parent); x1, y1 = pos(child)
        _draw_arrow(ax, x0, y0, x1, y1, color, 1.3, node_r)
        if show_latency and lat is not None:
            d = math.hypot(x1-x0, y1-y0)
            ux, uy = (x1-x0)/d, (y1-y0)/d
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx - uy*0.18, my + ux*0.18, f"{lat:.1f} µs",
                    fontsize=lat_fs, color=color, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.14", fc=bg, ec="none", alpha=0.9),
                    zorder=4)

    # nodes
    for r in ranks:
        x, y = pos(r)
        nc = root_color if r == source else node_color
        ax.add_patch(plt.Circle((x, y), node_r, color=nc,
                                zorder=5, linewidth=1.4, ec=node_edge))
        ax.text(x, y, str(r), ha="center", va="center",
                fontsize=node_fs, fontweight="600", color="white", zorder=6)


# ── MODE: grid ────────────────────────────────────────────────────────────────

def plot_grid(ops, cfg):
    ranks, steps = summarize(ops)
    exchanges    = build_exchanges(ops)
    n            = len(ranks)
    p            = auto_grid_params(n, len(steps), cfg.grid_cols)
    palette      = cfg.palette

    figsize = cfg.figsize or (p["fig_w"], p["fig_h"])
    dpi     = cfg.dpi     or _auto_dpi(*figsize)

    fig, ax = plt.subplots(figsize=figsize, facecolor=cfg.bg)
    ax.set_facecolor(cfg.bg); ax.axis("off")
    ax.set_title(cfg.title or f"Allgather forest · {n} ranks · {len(steps)} steps",
                 fontsize=p["title_fs"], color="#444444", pad=14, fontweight="normal")

    for idx, source in enumerate(ranks):
        ri, ci = divmod(idx, p["ncols"])
        color   = palette[source % len(palette)]
        ox      = ci * p["cell_w"] + p["cell_w"] * 0.1
        oy      = -(ri * p["cell_h"])

        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (ox - p["cell_w"]*0.08, oy - p["cell_h"] + p["cell_h"]*0.08),
            p["cell_w"]*0.96, p["cell_h"]*0.88,
            boxstyle="round,pad=0.05",
            linewidth=0.8, edgecolor=color, facecolor=color,
            alpha=0.04, zorder=0))

        ax.text(ox + p["cell_w"]*0.38, oy + p["cell_h"]*0.04,
                f"source: rank {source}",
                fontsize=p["cell_title_fs"], color=color,
                ha="center", va="bottom", fontweight="500")

        draw_tree(ax, source, exchanges, ranks, steps,
                  ox=ox, oy=oy - p["row_h"]*0.3,
                  col_w=p["col_w"], row_h=p["row_h"], node_r=p["node_r"],
                  color=color, root_color=color,
                  node_color=cfg.node_color, node_edge=cfg.node_edge, bg=cfg.bg,
                  show_latency=cfg.show_latency,
                  lat_fs=p["lat_fs"], node_fs=p["node_fs"],
                  ghost_color=cfg.ghost_color)

    ax.autoscale()
    plt.tight_layout(pad=cfg.pad)
    _save_show(fig, cfg, dpi)


# ── MODE: overlay ─────────────────────────────────────────────────────────────

def plot_overlay(ops, cfg):
    ranks, steps = summarize(ops)
    exchanges    = build_exchanges(ops)
    n            = len(ranks)
    ns           = len(steps)
    p            = auto_overlay_params(n, ns)
    palette      = cfg.palette

    figsize = cfg.figsize or (p["fig_w"], p["fig_h"])
    dpi     = cfg.dpi     or _auto_dpi(*figsize)

    fig, ax = plt.subplots(figsize=figsize, facecolor=cfg.bg)
    ax.set_facecolor(cfg.bg); ax.axis("off")
    ax.set_title(cfg.title or f"Allgather forest (overlay) · {n} ranks · {ns} steps",
                 fontsize=13, color="#444444", pad=14, fontweight="normal")

    rank_x = {r: r * p["col_w"] for r in ranks}

    # top rank labels
    for r in ranks:
        ax.text(rank_x[r], p["row_h"]*0.6, f"rank {r}",
                fontsize=p["node_fs"], color="#999999",
                ha="center", va="bottom", style="italic")

    for idx, source in enumerate(ranks):
        color  = palette[source % len(palette)]
        base_y = -idx * (ns + 1.8) * p["row_h"]

        _, tree_edges = build_broadcast_tree(source, exchanges, ranks)
        node_step_map = {source: -1}
        for parent, child, step, _ in tree_edges:
            node_step_map[child] = step

        def node_pos(r, _nsm=node_step_map):
            s   = _nsm.get(r, ns)
            row = 0 if s == -1 else s + 1
            return rank_x[r], base_y - row * p["row_h"]

        # ghost vertical lanes
        for r in ranks:
            x, _ = node_pos(r)
            ax.plot([x, x],
                    [base_y - (ns+1)*p["row_h"], base_y + p["row_h"]*0.3],
                    color=cfg.ghost_color, lw=0.4,
                    linestyle=(0, (4, 6)), zorder=0)

        # source label
        ax.text(rank_x[ranks[0]] - p["col_w"]*0.6, base_y,
                f"src {source}", fontsize=p["node_fs"],
                color=color, ha="right", va="center", fontweight="500")

        # arrows + latency
        for parent, child, step, lat in tree_edges:
            x0, y0 = node_pos(parent); x1, y1 = node_pos(child)
            _draw_arrow(ax, x0, y0, x1, y1, color, 1.3, p["node_r"])
            if cfg.show_latency and lat is not None:
                d = math.hypot(x1-x0, y1-y0)
                ux, uy = (x1-x0)/d, (y1-y0)/d
                mx, my = (x0+x1)/2, (y0+y1)/2
                ax.text(mx - uy*0.15, my + ux*0.15, f"{lat:.1f}µs",
                        fontsize=p["lat_fs"], color=color,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc=cfg.bg,
                                  ec="none", alpha=0.9), zorder=4)

        # nodes
        for r in ranks:
            x, y = node_pos(r)
            nc   = color if r == source else cfg.node_color
            ax.add_patch(plt.Circle((x, y), p["node_r"], color=nc,
                                    zorder=5, linewidth=1.4, ec=cfg.node_edge))
            ax.text(x, y, str(r), ha="center", va="center",
                    fontsize=p["node_fs"], fontweight="600",
                    color="white", zorder=6)

    if cfg.show_legend:
        patches = [mpatches.Patch(color=palette[r % len(palette)],
                                  label=f"Source {r}") for r in ranks]
        ax.legend(handles=patches, loc=cfg.legend_loc,
                  fontsize=7.5, title="Source rank", title_fontsize=8.5,
                  framealpha=0.0, edgecolor="none", labelcolor="#555555",
                  ncol=max(1, n // 8))

    ax.autoscale()
    plt.tight_layout(pad=cfg.pad)
    _save_show(fig, cfg, dpi)


# ── MODE: butterfly ───────────────────────────────────────────────────────────

def plot_butterfly(ops, cfg):
    ranks, steps = summarize(ops)
    exchanges    = build_exchanges(ops)
    n            = len(ranks)
    ns           = len(steps)
    p            = auto_butterfly_params(n, ns)
    palette      = cfg.palette

    figsize = cfg.figsize or (p["fig_w"], p["fig_h"])
    dpi     = cfg.dpi     or _auto_dpi(*figsize)

    fig, ax = plt.subplots(figsize=figsize, facecolor=cfg.bg)
    ax.set_facecolor(cfg.bg); ax.axis("off")
    ax.set_title(cfg.title or f"Butterfly · {n} ranks · {ns} steps",
                 fontsize=13, color="#444444", pad=16, fontweight="normal")

    rank_y = {r: -i * p["row_h"] for i, r in enumerate(ranks)}
    step_x = {s:  i * p["col_w"] for i, s in enumerate(steps)}

    def step_color(s):
        return palette[s % len(palette)]

    node_r  = p["node_r"]
    jitter  = node_r * 0.55   # horizontal offset so send/recv don't overlap

    # ghost lanes
    x_min = step_x[steps[0]]  - p["col_w"] * 0.4
    x_max = step_x[steps[-1]] + p["col_w"] * 0.4
    y_min = rank_y[ranks[-1]] - p["row_h"] * 0.4
    y_max = rank_y[ranks[0]]  + p["row_h"] * 0.4
    for r in ranks:
        ax.plot([x_min, x_max], [rank_y[r]]*2,
                color=cfg.ghost_color, lw=0.6,
                linestyle=(0, (6, 6)), zorder=0)
    for s in steps:
        ax.plot([step_x[s]]*2, [y_min, y_max],
                color=cfg.ghost_color, lw=0.4,
                linestyle=(0, (3, 8)), zorder=0)

    # arrows
    drawn = set()
    for ex in exchanges:
        ra, rb = ex["rank_a"], ex["rank_b"]
        s      = ex["step"]
        key    = (ra, rb, s)
        if key in drawn:
            continue
        drawn.add(key)

        c  = step_color(s)
        x  = step_x[s]
        ya = rank_y[ra]
        yb = rank_y[rb]
        if abs(yb - ya) < 1e-9:
            continue

        # send: ra → rb (left side of column)
        _draw_arrow(ax, x - jitter, ya, x - jitter, yb, c, cfg.arrow_lw, node_r)
        # recv: rb → ra (right side of column)
        _draw_arrow(ax, x + jitter, yb, x + jitter, ya, c, cfg.arrow_lw, node_r)

        if cfg.show_latency:
            mid_y = (ya + yb) / 2.0
            if ex["send_lat"] is not None:
                ax.text(x - jitter - node_r*0.8, mid_y,
                        f"{ex['send_lat']:.1f} µs",
                        fontsize=p["lat_fs"], color=c,
                        ha="right", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=cfg.bg,
                                  ec="none", alpha=0.85), zorder=3)
            if ex["recv_lat"] is not None:
                ax.text(x + jitter + node_r*0.8, mid_y,
                        f"{ex['recv_lat']:.1f} µs",
                        fontsize=p["lat_fs"], color=c,
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=cfg.bg,
                                  ec="none", alpha=0.85), zorder=3)

    # nodes
    for s in steps:
        x = step_x[s]
        for r in ranks:
            y = rank_y[r]
            ax.add_patch(plt.Circle((x, y), node_r, color=cfg.node_color,
                                    zorder=4, linewidth=1.4, ec=cfg.node_edge))
            ax.text(x, y, str(r), ha="center", va="center",
                    fontsize=p["node_fs"], fontweight="600",
                    color="white", zorder=5)

    # rank labels
    x_lbl = step_x[steps[0]] - p["col_w"] * 0.55
    for r in ranks:
        ax.text(x_lbl, rank_y[r], f"rank {r}",
                fontsize=p["rank_fs"], color="#888888",
                ha="right", va="center", style="italic")

    # step labels
    y_lbl = rank_y[ranks[0]] + p["row_h"] * 0.55
    for s in steps:
        ax.text(step_x[s], y_lbl, f"step {s}",
                fontsize=p["step_fs"], color="#888888",
                ha="center", va="bottom", style="italic")

    # legend
    if cfg.show_legend:
        patches = [mpatches.Patch(color=step_color(s), label=f"Step {s}")
                   for s in steps]
        ax.legend(handles=patches, loc=cfg.legend_loc,
                  fontsize=8.5, title="Step", title_fontsize=9,
                  framealpha=0.0, edgecolor="none", labelcolor="#555555")

    ax.autoscale()
    plt.tight_layout(pad=cfg.pad)
    _save_show(fig, cfg, dpi)


# ── save/show ─────────────────────────────────────────────────────────────────

def _save_show(fig, cfg, dpi):
    if not cfg.no_save:
        plt.savefig(cfg.out, dpi=dpi, facecolor=cfg.bg, bbox_inches="tight")
        print(f"Saved {cfg.out}  (dpi={dpi})")
    if not cfg.no_show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("json_file")

    p.add_argument("--mode", choices=["grid", "overlay", "butterfly"],
                   default="grid",
                   help="Visualization mode (default: grid)")

    # output
    p.add_argument("--out",     default=None,
                   help="Output filename (default: collbench_<mode>.png)")
    p.add_argument("--dpi",     type=int, default=None,
                   help="Override auto DPI")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-show", action="store_true")

    # layout overrides (all optional; auto-computed when absent)
    p.add_argument("--figsize",    type=float, nargs=2, default=None,
                   metavar=("W", "H"), help="Override figure size in inches")
    p.add_argument("--col-w",      type=float, default=None)
    p.add_argument("--row-h",      type=float, default=None)
    p.add_argument("--node-r",     type=float, default=None)
    p.add_argument("--pad",        type=float, default=1.3)
    p.add_argument("--grid-cols",  type=int,   default=None,
                   help="Grid mode: number of columns")
    p.add_argument("--arrow-lw",   type=float, default=1.4)

    # appearance
    p.add_argument("--bg",          default="#FAFAFA")
    p.add_argument("--node-color",  default="#3A7BD5")
    p.add_argument("--node-edge",   default="#FFFFFF")
    p.add_argument("--ghost-color", default="#DDDDDD")
    p.add_argument("--palette",     default=None,
                   help="Comma-separated hex colors")

    # labels
    p.add_argument("--title",       default=None)

    # toggles
    p.add_argument("--no-latency",  action="store_true")
    p.add_argument("--no-legend",   action="store_true")
    p.add_argument("--legend-loc",  default="upper right")

    return p.parse_args()


def main():
    args    = parse_args()
    ops     = load_ops(args.json_file)
    palette = ([c.strip() for c in args.palette.split(",")]
               if args.palette else DEFAULT_PALETTE)

    out = args.out or f"collbench_{args.mode}.png"

    cfg = argparse.Namespace(
        mode         = args.mode,
        out          = out,
        dpi          = args.dpi,
        no_save      = args.no_save,
        no_show      = args.no_show,
        figsize      = tuple(args.figsize) if args.figsize else None,
        grid_cols    = args.grid_cols,
        pad          = args.pad,
        arrow_lw     = args.arrow_lw,
        bg           = args.bg,
        node_color   = args.node_color,
        node_edge    = args.node_edge,
        ghost_color  = args.ghost_color,
        palette      = palette,
        title        = args.title,
        show_latency = not args.no_latency,
        show_legend  = not args.no_legend,
        legend_loc   = args.legend_loc,
    )

    dispatch = dict(grid=plot_grid, overlay=plot_overlay, butterfly=plot_butterfly)
    dispatch[cfg.mode](ops, cfg)


if __name__ == "__main__":
    main()
