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
    tunnel_params = TunnelParams(
        {
            "distance": 400,
            "starting_direction": np.array((1, 0, 0)),
            "horizontal_tendency": np.deg2rad(10),
            "horizontal_noise": np.deg2rad(30),
            "vertical_tendency": np.deg2rad(00),
            "vertical_noise": np.deg2rad(2),
            "min_seg_length": 40,
            "max_seg_length": 50,
        }
    )
    # Generate the graph
    graph = TunnelNetwork()
    Node.set_graph(graph)
    Tunnel(graph, np.array((0, 0, 0)), tunnel_params)
    debug_plot(graph)
    node = list(list(graph._tunnels)[0]._nodes)[2]
    for th in np.linspace(0, 2 * np.pi, 5)[:-1]:
        starting_direction = angles_to_vector((th, 0))
        print(starting_direction)
        params = TunnelParams({"starting_direction": starting_direction})
        Tunnel(graph, node, params=params)
        debug_plot(graph)


if __name__ == "__main__":
    main()
