import numpy as np
import matplotlib.pyplot as plt

from helper_functions import *

from tunnel import TunnelParams, Tunnel, TunnelNetwork
from graph import Node
from display_functions import plot_graph_2d, network_overview
from mesh_generation import get_mesh_vertices_from_graph_perlin_and_spline, mesh_from_vertices, plot_mesh


def debug_plot(graph):
    assert isinstance(graph, TunnelNetwork)
    plt.gca().clear()
    plot_graph_2d(graph)
    plt.draw()
    plt.waitforbuttonpress()


def main():
    fig = plt.figure(figsize=(8, 8))
    axis = plt.subplot(1, 1, 1)
    plt.show(block=False)
    tunnel_params = TunnelParams({"distance": 100,
                                  "starting_direction": np.array((1, 0, 0)),
                                  "horizontal_tendency": np.deg2rad(0),
                                  "horizontal_noise": np.deg2rad(20),
                                  "vertical_tendency": np.deg2rad(10),
                                  "vertical_noise": np.deg2rad(5),
                                  "min_seg_length": 20,
                                  "max_seg_length": 30})
    # Generate the graph
    graph = TunnelNetwork()
    Node.set_graph(graph)
    Tunnel(graph,np.array((0, 0, 0)), tunnel_params)
    node = graph.nodes[-3]
    tunnel_params["starting_direction"] = np.array((0,1,0))
    #Tunnel(graph, node, tunnel_params)
    #tunnel_params["starting_direction"] = np.array((0,-1,0))
    #Tunnel(graph, node, tunnel_params)
    #plot_graph_2d(graph)
    # Generate the vertices of the mesh
    print("Generating mesh")
    points, normals = get_mesh_vertices_from_graph_perlin_and_spline(graph)
    mesh, ptcl = mesh_from_vertices(points, normals)
    plot_mesh(ptcl)
    input()

         
if __name__ == "__main__":
    main()
