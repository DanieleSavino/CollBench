import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import matplotlib.font_manager as fm
from collections import defaultdict
import colorsys
import math

def best_sans():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ["DejaVu Sans", "Liberation Sans", "FreeSans",
                      "Noto Sans", "Ubuntu", "Cantarell"]:
        if candidate in available:
            return candidate
    return "sans-serif"

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": [best_sans()],
    "axes.spines.left":   False,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.spines.bottom": False,
})

PALETTE = [
    "#E05C5C",  # round 0  warm red
    "#5B8FF9",  # round 1  blue
    "#5AD8A6",  # round 2  teal
    "#F6BD16",  # round 3  amber
    "#9370DB",  # round 4  purple
]
ROOT_COLOR  = "#E05C5C"
NODE_COLOR  = "#3A7BD5"
NODE_EDGE   = "#FFFFFF"
GHOST_COLOR = "#DDDDDD"
BG_COLOR    = "#FAFAFA"


def build_edges(ops):
    return [
        {"src": o["rank"], "dst": o["peer"],
         "round": o["algo_idx"], "lat": o["t_total_ns"] / 1000.0}
        for o in ops if o["operation_type"] == "send"
    ]


def assign_layout(edges, root):
    children   = defaultdict(list)
    first_round = {root: -1}
    parent_of  = {}

    for e in sorted(edges, key=lambda x: x["round"]):
        s, d, r = e["src"], e["dst"], e["round"]
        if d not in first_round:
            first_round[d] = r
            parent_of[d]   = s
        if first_round.get(s, -1) < r:
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

    row = {n: first_round[n] + 1 for n in first_round}
    row[root] = 0
    return node_col, row, children, parent_of, first_round


def round_color(r):
    return PALETTE[r % len(PALETTE)]


def plot(ops, root):
    edges = build_edges(ops)
    col, row, children, parent_of, first_round = assign_layout(edges, root)

    all_nodes = set(col.keys())
    n_rounds  = max(row.values()) + 1

    COL_W, ROW_H = 2.6, 2.2
    pos = {n: (col[n] * COL_W, -row[n] * ROW_H) for n in all_nodes}

    n_cols = max(col.values()) - min(col.values()) + 1

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")
    ax.set_title(f"Tree {len(all_nodes)} ranks  ·  root = {root}",
                 fontsize=12, color="#444444", pad=16, fontweight="normal")

    lat_map = {(e["src"], e["dst"]): e["lat"]   for e in edges}
    rnd_map = {(e["src"], e["dst"]): e["round"]  for e in edges}

    NODE_R = 0.30

    # ── dotted skeleton ──────────────────────────────────────────────────────
    for src, dsts in children.items():
        for dst in dsts:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            ax.plot([x0, x1], [y0, y1],
                    color=GHOST_COLOR, lw=0.8,
                    linestyle=(0, (4, 5)), zorder=0, solid_capstyle="round")

    # ── colored arrows ───────────────────────────────────────────────────────
    for e in edges:
        s, d = e["src"], e["dst"]
        r    = rnd_map[(s, d)]
        lat  = lat_map[(s, d)]
        c    = round_color(r)

        x0, y0 = pos[s]
        x1, y1 = pos[d]
        dx, dy  = x1 - x0, y1 - y0
        dist    = math.hypot(dx, dy)
        ux, uy  = dx / dist, dy / dist

        xs, ys = x0 + ux * NODE_R, y0 + uy * NODE_R
        xe, ye = x1 - ux * NODE_R, y1 - uy * NODE_R

        ax.annotate("",
            xy=(xe, ye), xytext=(xs, ys),
            arrowprops=dict(
                arrowstyle="-|>",
                color=c,
                lw=1.5,
                mutation_scale=12,
                shrinkA=0, shrinkB=0,
            ),
            zorder=2,
        )

        # latency label — offset perpendicular to edge
        mx = (xs + xe) / 2 - uy * 0.22
        my = (ys + ye) / 2 + ux * 0.22
        ax.text(mx, my, f"{lat:.1f} µs",
                fontsize=7.5, color=c, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc=BG_COLOR,
                          ec="none", alpha=0.9),
                zorder=3)

    # ── nodes ────────────────────────────────────────────────────────────────
    for n in all_nodes:
        x, y  = pos[n]
        color = ROOT_COLOR if n == root else NODE_COLOR
        circ  = plt.Circle((x, y), NODE_R,
                            color=color, zorder=4, linewidth=1.8,
                            ec=NODE_EDGE)
        ax.add_patch(circ)
        ax.text(x, y, str(n),
                ha="center", va="center",
                fontsize=10, fontweight="600",
                color="white", zorder=5)

    # ── round labels (left margin) ───────────────────────────────────────────
    for r_i in range(n_rounds):
        nodes_in_row = [n for n in all_nodes if row[n] == r_i]
        if not nodes_in_row:
            continue
        min_x = min(pos[n][0] for n in nodes_in_row)
        y     = -r_i * ROW_H
        label = "root" if r_i == 0 else f"round {r_i - 1}"
        ax.text(min_x - 0.55, y, label,
                fontsize=8, color="#AAAAAA",
                ha="right", va="center", style="italic")

    # ── legend ───────────────────────────────────────────────────────────────
    actual_rounds = sorted({e["round"] for e in edges})
    patches = [mpatches.Patch(color=round_color(r), label=f"Round {r}")
               for r in actual_rounds]
    patches.append(mpatches.Patch(color=ROOT_COLOR, label=f"Root ({root})"))
    leg = ax.legend(handles=patches, loc="upper right",
                    fontsize=8.5, title="Round",
                    title_fontsize=9, framealpha=0.0,
                    edgecolor="none", labelcolor="#555555")

    ax.autoscale()
    plt.tight_layout(pad=1.4)
    plt.savefig("collbench_tree.png", dpi=220,
                facecolor=BG_COLOR, bbox_inches="tight")
    plt.show()
    print("Saved collbench_tree.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "out.json"
    root = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(path) as f:
        data = json.load(f)
    plot(data["operations"], root)
