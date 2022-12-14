import numpy as np
import matplotlib.pyplot as plt

from helper_functions import *

from tunnel import TunnelParams, Tunnel, TunnelNetwork
from graph import Node
from display_functions import plot_graph_2d, network_overview
from mesh_generation import *
import open3d as o3d
COLORS = (
    np.array((255,255,0))/255.,
    np.array((128,128,0))/255.,
    np.array((0,0,255))/255.,
    np.array((75,0,130))/255.,
    np.array((75,0,130))/255.,
    np.array((255,69,0))/255.,
)
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
    Tunnel(graph, node, tunnel_params)
    #tunnel_params["starting_direction"] = np.array((0,-1,0))
    #Tunnel(graph, node, tunnel_params)
    plot_graph_2d(graph, axis)
    plt.draw()
    plt.waitforbuttonpress()
    # Generate the vertices of the mesh
    print("Generating mesh")
    points, normals = get_vertices_for_tunnels(graph)
    print(points)
    meshes, ptcls = list(), list()
    # Get the mesh for each tunnel
    for i, (p, n) in enumerate(zip(points, normals)):
        ptcl = o3d.geometry.PointCloud()
        ptcl.points = o3d.utility.Vector3dVector(p.T)
        ptcl.normals = o3d.utility.Vector3dVector(n.T)
        ptcl.colors = o3d.utility.Vector3dVector(np.ones(np.asarray(ptcl.points).shape)*COLORS[i])
        ptcls.append(ptcl)
    axis_ptcls = list()
    for tunnel in graph._tunnels:
        axis_ptcls.append(get_axis_pointcloud(tunnel))
    o3d.visualization.draw_geometries(ptcls+axis_ptcls)
    input()

         
if __name__ == "__main__":
    main()
