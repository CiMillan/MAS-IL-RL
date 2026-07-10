from __future__ import annotations
import networkx as nx


def build_square_lattice_graph(rows: int, cols: int) -> nx.Graph:
    g = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def neighbor_lists_from_graph(graph: nx.Graph):
    return [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]
