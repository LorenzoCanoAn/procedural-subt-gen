import numpy as np
import matplotlib.pyplot as plt

from helper_functions import *
import pickle
from tunnel import TunnelParams, Tunnel, TunnelNetwork
from graph import Node
from display_functions import plot_graph_2d, network_overview
from mesh_generation import *
import open3d as o3d

def debug_plot(graph):
    assert isinstance(graph, TunnelNetwork)
    plt.gca().clear()
    plot_graph_2d(graph)
    plt.draw()
    plt.waitforbuttonpress()


def main():
    tunnel_params = TunnelParams({"distance": 200,
                                  "starting_direction": np.array((1, 0, 0)),
                                  "horizontal_tendency": np.deg2rad(20),
                                  "horizontal_noise": np.deg2rad(10),
                                  "vertical_tendency": np.deg2rad(20),
                                  "vertical_noise": np.deg2rad(5),
                                  "min_seg_length": 10,
                                  "max_seg_length": 20})
    # Generate the graph
    graph = TunnelNetwork()
    Node.set_graph(graph)
    Tunnel(graph,np.array((0, 0, 0)), tunnel_params)
    first_tunnel_nodes = list(list(graph._tunnels)[0]._nodes)
    tunnel_params["starting_direction"] = np.array((0,1,0))
    Tunnel(graph, first_tunnel_nodes[1], tunnel_params)
    tunnel_params["starting_direction"] = np.array((0,-1,0))
    debug_plot(graph)
    Tunnel(graph, first_tunnel_nodes[2], tunnel_params)
    debug_plot(graph)
    Tunnel(graph, first_tunnel_nodes[3], tunnel_params)
    debug_plot(graph)
    Tunnel(graph, first_tunnel_nodes[4], tunnel_params)
    debug_plot(graph)
    Tunnel(graph, first_tunnel_nodes[5], tunnel_params)
    debug_plot(graph)
    Tunnel(graph, first_tunnel_nodes[6], tunnel_params)
    debug_plot(graph)
    
    with open("graph.pkl", "wb") as f:
        pickle.dump(graph, f)
         
if __name__ == "__main__":
    main()
