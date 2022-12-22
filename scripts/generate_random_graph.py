import numpy as np
import matplotlib.pyplot as plt

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import plot_graph_2d, network_overview
from subt_proc_gen.mesh_generation import *
import open3d as o3d


def debug_plot(graph):
    assert isinstance(graph, TunnelNetwork)
    plt.gca().clear()
    plot_graph_2d(graph)
    plt.draw()
    plt.waitforbuttonpress()


def main():
    for i in range(1):
        tunnel_params = TunnelParams(
            {
                "distance": 800,
                "starting_direction": np.array((1, 0, 0)),
                "horizontal_tendency": np.deg2rad(10),
                "horizontal_noise": np.deg2rad(30),
                "vertical_tendency": np.deg2rad(00),
                "vertical_noise": np.deg2rad(20),
                "min_seg_length": 40,
                "max_seg_length": 50,
            }
        )
        # Generate the graph
        graph = TunnelNetwork()
        central_node = CaveNode()
        graph.add_node(central_node)
        for th in np.random.uniform(0, 2 * np.pi, 6):
            ph = np.random.uniform(-20, 20)
            ph = np.deg2rad(ph)
            starting_direction = angles_to_vector((th, ph))
            tunnel_params["starting_direction"] = starting_direction
            node = random.choice(graph.nodes)
            Tunnel(graph, node, params=tunnel_params)
        debug_plot(graph)
        with open("datafiles/graph_.pkl", "wb") as f:
            pickle.dump(graph, f)


if __name__ == "__main__":
    main()
