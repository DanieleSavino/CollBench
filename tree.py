import json
import sys
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

def rank2nb(rank, s):
    return rank ^ (rank >> 1)

def nb2rank(nb, s):
    rank = 0
    while nb:
        rank ^= nb
        nb >>= 1
    return rank

def build_graph(ops, size, root=0):
    s = math.ceil(math.log2(size)) if size > 1 else 1
    G = nx.DiGraph()
    for r in range(size):
        G.add_node(r)

    for op in ops:
        if op["operation_type"] != "send":
            continue

        r      = op["rank"]
        round_i = op["algo_idx"]
        nb_r   = rank2nb(r, s)
        mask   = ((1 << s) - 1) >> round_i
        nb_q   = nb_r ^ mask
        q      = nb2rank(nb_q, s)

        if q >= size:
            continue

        G.add_edge(r, q, round=round_i,
                   latency_us=op["t_total_ns"] / 1000.0)

    return G

def hierarchy_pos(G, root, size):
    depth = {root: 0}
    for u, v, d in G.edges(data=True):
        depth[v] = d["round"] + 1
    for node in G.nodes():
        if node not in depth:
            depth[node] = 0

    levels = defaultdict(list)
    for node, d in depth.items():
        levels[d].append(node)

    pos = {}
    for d, nodes in levels.items():
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = ((i - (n - 1) / 2.0) * 2.0, -d * 2.0)
    return pos

def plot(ops, size, root=0):
    G   = build_graph(ops, size, root)
    pos = hierarchy_pos(G, root, size)

    n_rounds = max((d["round"] for _, _, d in G.edges(data=True)), default=0) + 1
    cmap     = plt.cm.tab10
    colors   = [cmap(i / max(n_rounds - 1, 1)) for i in range(n_rounds)]

    fig, ax = plt.subplots(figsize=(max(8, size * 1.5), max(5, n_rounds * 3)))
    ax.set_title(f"BINE bcast tree  —  {size} ranks, root={root}", fontsize=13)
    ax.axis("off")

    for round_i in range(n_rounds):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["round"] == round_i]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            edge_color=[colors[round_i]] * len(edges),
            arrows=True,
            arrowstyle="-|>",
            arrowsize=25,
            width=2.5,
            node_size=900,
            connectionstyle="arc3,rad=0.0",
        )

    edge_labels = {
        (u, v): f"{d['latency_us']:.1f}µs"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=8, ax=ax,
                                  bbox=dict(boxstyle="round,pad=0.2",
                                            fc="white", ec="none", alpha=0.8))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900, node_color="steelblue")
    nx.draw_networkx_labels(G, pos, {r: str(r) for r in range(size)},
                             ax=ax, font_color="white", font_weight="bold", font_size=10)

    patches = [mpatches.Patch(color=colors[i], label=f"Round {i}")
               for i in range(n_rounds)]
    ax.legend(handles=patches, loc="upper right", fontsize=9, title="Round")
    plt.tight_layout()
    plt.savefig("collbench_tree.png", dpi=150)
    plt.show()
    print("Saved collbench_tree.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "out.json"
    root = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(path) as f:
        data = json.load(f)
    ops  = data["operations"]
    size = max(o["rank"] for o in ops) + 1
    plot(ops, size, root)
