"""Visualise graph(s)."""

import argparse

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("FILE", type=str, help="Input file (in `.g6` format)")

    args = parser.parse_args()
    graphs = nx.read_graph6(args.FILE)

    n = len(graphs)
    n_rows = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_rows, squeeze=False)

    for ax in axes.ravel():
        ax.set_visible(False)

    for graph, ax in zip(graphs, axes.ravel()):
        ax.axis("equal")
        ax.set_visible(True)
        ax.set_box_aspect(1)

        pos = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(
            graph,
            pos=pos,
            node_color="black",
            ax=ax,
            node_size=50,
            with_labels=False,
        )

    plt.show()
