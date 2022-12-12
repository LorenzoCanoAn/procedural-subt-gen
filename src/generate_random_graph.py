import numpy as np
import matplotlib.pyplot as plt

from helper_functions import *

from tunnel import TunnelParams
from graph import Graph, Node
from display_functions import plot_graph_2d

def debug_plot(graph):
    assert isinstance(graph, Graph)
    plt.gca().clear()
    plot_graph_2d(graph)
    plt.draw()
    plt.waitforbuttonpress()


def main():
    n_rows = 5
    n_cols = 5
    fig = plt.figure(figsize=(8, 8))
    axis = plt.subplot(1, 1, 1)
    plt.show(block=False)
    while True:
        tunnel_params = TunnelParams({"distance": 100,
                                      "starting_direction": np.array((1, 0, 0)),
                                      "horizontal_tendency": np.deg2rad(0),
                                      "horizontal_noise": np.deg2rad(20),
                                      "vertical_tendency": np.deg2rad(10),
                                      "vertical_noise": np.deg2rad(5),
                                      "min_seg_length": 20,
                                      "max_seg_length": 30})
        graph = Graph()
        Node.set_graph(graph)  # This is so all nodes share the same graph
        graph.add_floating_tunnel(np.array((0, 0, 0)), tunnel_params)
        debug_plot(graph)
        node = graph.nodes[-3]
        graph.add_tunnel(node, tunnel_params)
        debug_plot(graph)
        tunnel_params["starting_direction"] = np.array((0, 1, 0))
        graph.add_tunnel(node, tunnel_params)
        debug_plot(graph)


if __name__ == "__main__":
    main()
