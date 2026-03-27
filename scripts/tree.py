#!/usr/bin/env python3
"""
collbench_tree.py — visualize a broadcast/collective communication tree
from a CollBench JSON export.

Supports both root-to-leaves (broadcast) and leaves-to-root (gather/reduce)
algorithms.  Direction is auto-detected but can be overridden.

Usage:
    python collbench_tree.py [OPTIONS] <json_file>

Examples:
    python collbench_tree.py out/bine_bcast_dhlv.json
    python collbench_tree.py --root 3 --dpi 300 --out tree.png out.json
    python collbench_tree.py --no-latency --col-w 3.0 --row-h 2.5 out.json
    python collbench_tree.py --palette "#E05C5C,#5B8FF9,#5AD8A6" out.json
    python collbench_tree.py --direction gather out/gather.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ── font ──────────────────────────────────────────────────────────────────────

def best_sans():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ["DejaVu Sans", "Liberation Sans", "FreeSans",
                      "Noto Sans", "Ubuntu", "Cantarell"]:
        if candidate in available:
            return candidate
    return "sans-serif"


matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    [best_sans()],
    "axes.spines.left":   False,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.spines.bottom": False,
})

DEFAULT_PALETTE = [
    "#E05C5C",
    "#5B8FF9",
    "#5AD8A6",
    "#F6BD16",
    "#9370DB",
]


# ── direction detection ───────────────────────────────────────────────────────

def detect_direction(ops, root):
    """
    If root only receives → gather. If root only sends → broadcast.
    Fallback: global send/recv majority.
    """
    root_sends = sum(1 for o in ops if o["rank"] == root and o["operation_type"] == "send")
    root_recvs = sum(1 for o in ops if o["rank"] == root and o["operation_type"] == "recv")
    if root_recvs > 0 and root_sends == 0:
        return "gather"
    if root_sends > 0 and root_recvs == 0:
        return "broadcast"
    total_sends = sum(1 for o in ops if o["operation_type"] == "send")
    total_recvs = sum(1 for o in ops if o["operation_type"] == "recv")
    return "gather" if total_recvs > total_sends else "broadcast"


# ── data helpers ──────────────────────────────────────────────────────────────

def build_edges(ops, direction):
    if direction == "broadcast":
        # src=parent (sender), dst=child — layout and arrow both parent→child
        return [
            {"src": o["rank"], "dst": o["peer"],
             "round": o["algo_idx"], "lat": o["t_total_ns"] / 1000.0}
            for o in ops if o["operation_type"] == "send"
        ]
    else:  # gather: use send ops, rank=child (sender), peer=parent (receiver)
        # For layout: swap so src=parent, dst=child (assign_layout expects parent=src)
        # arrow_src/arrow_dst keep original direction: child→parent (leaf→root)
        return [
            {"src": o["peer"], "dst": o["rank"],   # layout: parent→child
             "round": o["algo_idx"], "lat": o["t_total_ns"] / 1000.0,
             "arrow_src": o["rank"], "arrow_dst": o["peer"]}  # draw: child→parent
            for o in ops if o["operation_type"] == "send"
        ]


def assign_layout(edges, root, direction="broadcast"):
    """
    Assign (col, row) positions.
    broadcast: src=parent, dst=child; earlier round = shallower row.
    gather:    src=parent (root-ward), dst=child (leaf-ward); we use the
               send-round of each node to assign depth (leaves send first
               so they go deepest; root never sends so it stays at row 0).
    """
    children    = defaultdict(list)
    first_round = {root: -1}
    parent_of   = {}

    for e in sorted(edges, key=lambda x: x["round"]):
        s, d, r = e["src"], e["dst"], e["round"]
        if d not in first_round:
            first_round[d] = r
            parent_of[d]   = s
        if first_round.get(s, -1) < r:
            if d not in children[s]:
                children[s].append(d)

    col_counter = [0]
    node_col    = {}

    def fill_col(node):
        kids = children.get(node, [])
        if not kids:
            node_col[node] = col_counter[0]
            col_counter[0] += 1
            return
        for k in kids:
            fill_col(k)
        cols = [node_col[k] for k in kids]
        node_col[node] = (cols[0] + cols[-1]) / 2.0

    fill_col(root)

    if direction == "gather":
        # Row = distance from root measured by how late a node sends.
        # Nodes that send early (small algo_idx) are deeper leaves.
        # Root never sends → row 0.  Deepest leaf → highest row number.
        send_round = {e["arrow_src"]: e["round"] for e in edges}
        max_r = max(send_round.values()) if send_round else 1
        row = {}
        for n in set(first_round.keys()):
            if n == root:
                row[n] = 0
            else:
                row[n] = max_r - send_round.get(n, 0) + 1
    else:
        row = {n: first_round[n] + 1 for n in first_round}
        row[root] = 0

    return node_col, row, children, parent_of, first_round


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(ops, cfg):
    root      = cfg.root
    palette   = cfg.palette
    direction = cfg.direction or detect_direction(ops, root)

    edges = build_edges(ops, direction)

    if not edges:
        sys.exit(
            f"[error] No edges found for direction='{direction}'. "
            "Try --direction broadcast or --direction gather."
        )

    col, row, children, parent_of, first_round = assign_layout(edges, root, direction)

    all_nodes = set(col.keys())
    n_rounds  = max(row.values()) + 1

    flip_h = -1 if direction == "gather" else 1

    pos = {
        n: (col[n] * cfg.col_w * flip_h, -row[n] * cfg.row_h)
        for n in all_nodes
    }

    fig, ax = plt.subplots(figsize=cfg.figsize, facecolor=cfg.bg)
    ax.set_facecolor(cfg.bg)
    ax.axis("off")

    direction_label = "Gather" if direction == "gather" else "Broadcast"
    if cfg.title:
        ax.set_title(cfg.title, fontsize=cfg.title_fontsize,
                     color="#444444", pad=16, fontweight="normal")
    else:
        ax.set_title(
            f"{direction_label}  ·  {len(all_nodes)} ranks  ·  root = {root}",
            fontsize=cfg.title_fontsize, color="#444444",
            pad=16, fontweight="normal",
        )

    def round_color(r):
        return palette[r % len(palette)]

    lat_map = {(e["src"], e["dst"]): e["lat"]   for e in edges}
    rnd_map = {(e["src"], e["dst"]): e["round"] for e in edges}

    NODE_R = cfg.node_r

    # ── skeleton ──────────────────────────────────────────────────────────────
    for src, dsts in children.items():
        for dst in dsts:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            ax.plot([x0, x1], [y0, y1],
                    color=cfg.ghost_color, lw=0.8,
                    linestyle=(0, (4, 5)), zorder=0,
                    solid_capstyle="round")

    # ── arrows ────────────────────────────────────────────────────────────────
    for e in edges:
        s, d = e["src"], e["dst"]
        r    = rnd_map[(s, d)]
        lat  = lat_map[(s, d)]
        c    = round_color(r)

        asrc = e.get("arrow_src", s)
        adst = e.get("arrow_dst", d)
        x0, y0 = pos[asrc]
        x1, y1 = pos[adst]
        dx, dy  = x1 - x0, y1 - y0
        dist    = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy  = dx / dist, dy / dist

        xs, ys = x0 + ux * NODE_R, y0 + uy * NODE_R
        xe, ye = x1 - ux * NODE_R, y1 - uy * NODE_R

        ax.annotate("",
            xy=(xe, ye), xytext=(xs, ys),
            arrowprops=dict(
                arrowstyle="-|>",
                color=c,
                lw=cfg.arrow_lw,
                mutation_scale=12,
                shrinkA=0, shrinkB=0,
            ),
            zorder=2,
        )

        if cfg.show_latency:
            mx = (xs + xe) / 2 - uy * 0.22
            my = (ys + ye) / 2 + ux * 0.22
            ax.text(mx, my, f"{lat:.1f} µs",
                    fontsize=cfg.lat_fontsize, color=c,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc=cfg.bg,
                              ec="none", alpha=0.9),
                    zorder=3)

    # ── nodes ─────────────────────────────────────────────────────────────────
    for n in all_nodes:
        x, y  = pos[n]
        color = cfg.root_color if n == root else cfg.node_color
        circ  = plt.Circle((x, y), NODE_R,
                            color=color, zorder=4,
                            linewidth=1.8, ec=cfg.node_edge)
        ax.add_patch(circ)
        ax.text(x, y, str(n),
                ha="center", va="center",
                fontsize=cfg.node_fontsize, fontweight="600",
                color="white", zorder=5)

    # ── row labels ────────────────────────────────────────────────────────────
    if cfg.show_row_labels:
        for r_i in range(n_rounds):
            nodes_in_row = [n for n in all_nodes if row[n] == r_i]
            if not nodes_in_row:
                continue
            min_x = min(pos[n][0] for n in nodes_in_row)
            y     = -r_i * cfg.row_h
            if direction == "gather":
                label = "root" if r_i == 0 else f"round {r_i - 1}"
            else:
                label = "root" if r_i == 0 else f"round {r_i - 1}"
            ax.text(min_x - 0.55, y, label,
                    fontsize=8, color="#AAAAAA",
                    ha="right", va="center", style="italic")

    # ── legend ────────────────────────────────────────────────────────────────
    if cfg.show_legend:
        actual_rounds = sorted({e["round"] for e in edges})
        patches = [
            mpatches.Patch(color=round_color(r), label=f"Round {r}")
            for r in actual_rounds
        ]
        patches.append(
            mpatches.Patch(color=cfg.root_color, label=f"Root ({root})")
        )
        ax.legend(handles=patches, loc=cfg.legend_loc,
                  fontsize=8.5, title="Round",
                  title_fontsize=9, framealpha=0.0,
                  edgecolor="none", labelcolor="#555555")

    ax.autoscale()
    plt.tight_layout(pad=cfg.pad)

    if not cfg.no_save:
        plt.savefig(cfg.out, dpi=cfg.dpi,
                    facecolor=cfg.bg, bbox_inches="tight")
        print(f"Saved {cfg.out}")

    if not cfg.no_show:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional
    p.add_argument("json_file", help="Path to CollBench JSON output")

    # core
    p.add_argument("--root", type=int, default=0,
                   help="Root rank (default: 0)")
    p.add_argument("--direction", choices=["broadcast", "gather"], default=None,
                   help="Collective direction: 'broadcast' (root→leaves) or "
                        "'gather' (leaves→root). Auto-detected if omitted.")

    # output
    p.add_argument("--out", default="collbench_tree.png",
                   help="Output image path (default: collbench_tree.png)")
    p.add_argument("--dpi", type=int, default=220,
                   help="Output DPI (default: 220)")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save to disk")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show()")

    # layout
    p.add_argument("--col-w", type=float, default=2.6,
                   help="Horizontal spacing between columns (default: 2.6)")
    p.add_argument("--row-h", type=float, default=2.2,
                   help="Vertical spacing between rows (default: 2.2)")
    p.add_argument("--figsize", type=float, nargs=2, default=[16, 9],
                   metavar=("W", "H"), help="Figure size in inches (default: 16 9)")
    p.add_argument("--pad", type=float, default=1.4,
                   help="tight_layout pad (default: 1.4)")
    p.add_argument("--node-r", type=float, default=0.30,
                   help="Node circle radius (default: 0.30)")

    # appearance
    p.add_argument("--bg", default="#FAFAFA",
                   help="Background color (default: #FAFAFA)")
    p.add_argument("--node-color", default="#3A7BD5",
                   help="Non-root node fill color (default: #3A7BD5)")
    p.add_argument("--root-color", default="#E05C5C",
                   help="Root node fill color (default: #E05C5C)")
    p.add_argument("--node-edge", default="#FFFFFF",
                   help="Node border color (default: #FFFFFF)")
    p.add_argument("--ghost-color", default="#DDDDDD",
                   help="Skeleton edge color (default: #DDDDDD)")
    p.add_argument("--palette", default=None,
                   help="Comma-separated hex colors for rounds "
                        "(default: built-in 5-color palette)")
    p.add_argument("--arrow-lw", type=float, default=1.5,
                   help="Arrow line width (default: 1.5)")

    # typography
    p.add_argument("--title", default=None,
                   help="Custom plot title (default: auto-generated)")
    p.add_argument("--title-fontsize", type=float, default=12,
                   help="Title font size (default: 12)")
    p.add_argument("--node-fontsize", type=float, default=10,
                   help="Node label font size (default: 10)")
    p.add_argument("--lat-fontsize", type=float, default=7.5,
                   help="Latency label font size (default: 7.5)")

    # toggles
    p.add_argument("--no-latency", action="store_true",
                   help="Hide latency labels on edges")
    p.add_argument("--no-legend", action="store_true",
                   help="Hide the round legend")
    p.add_argument("--no-row-labels", action="store_true",
                   help="Hide round/root row labels on the left")
    p.add_argument("--legend-loc", default="upper right",
                   help="Legend location (default: 'upper right')")

    return p.parse_args()


def main():
    args = parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    if args.palette:
        palette = [c.strip() for c in args.palette.split(",")]
    else:
        palette = DEFAULT_PALETTE

    cfg = argparse.Namespace(
        root           = args.root,
        direction      = args.direction,
        out            = args.out,
        dpi            = args.dpi,
        no_save        = args.no_save,
        no_show        = args.no_show,
        col_w          = args.col_w,
        row_h          = args.row_h,
        figsize        = tuple(args.figsize),
        pad            = args.pad,
        node_r         = args.node_r,
        bg             = args.bg,
        node_color     = args.node_color,
        root_color     = args.root_color,
        node_edge      = args.node_edge,
        ghost_color    = args.ghost_color,
        palette        = palette,
        arrow_lw       = args.arrow_lw,
        title          = args.title,
        title_fontsize = args.title_fontsize,
        node_fontsize  = args.node_fontsize,
        lat_fontsize   = args.lat_fontsize,
        show_latency   = not args.no_latency,
        show_legend    = not args.no_legend,
        show_row_labels= not args.no_row_labels,
        legend_loc     = args.legend_loc,
    )

    plot(data["operations"], cfg)


if __name__ == "__main__":
    main()
