"""Functions to display and debug the graph generation process"""
import matplotlib.pyplot as plt
import numpy as np
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork
from subt_proc_gen.PARAMS import *
import pyvista as pv


def plot_graph_2d(graph, ax=None):
    if ax is None:
        ax = plt.gca()

    for edge in graph.edges:
        edge.plot2d(ax)
    for node in graph.nodes:
        ax.scatter(node.x, node.y, c="b")
    for tunnel in graph._tunnels:
        plot_tunnel_2d(tunnel, ax)
    mincoords = np.array((graph.minx, graph.miny))
    maxcoords = np.array((graph.maxx, graph.maxy))
    max_diff = max(maxcoords - mincoords)
    ax.set_xlim(min(mincoords), max(maxcoords))
    ax.set_ylim(min(mincoords), max(maxcoords))


def plot_graph_3d(self, ax=None):
    if ax is None:
        ax = plt.gca()
    for edge in self.edges:
        edge.plot3d(ax)
    for node in self.nodes:
        ax.scatter3D(node.x, node.y, node.z, c="b")
    # for tunnel in self.tunnels:
    #    color = tuple(np.random.choice(range(256), size=3)/255)
    #    for i in range(len(tunnel)-1):
    #        x0 = tunnel[i].x
    #        x1 = tunnel[i+1].x
    #        y0 = tunnel[i].y
    #        y1 = tunnel[i+1].y
    #        z0 = tunnel[i].z
    #        z1 = tunnel[i+1].z
    #        ax.plot3D([x0, x1], [y0, y1], [z0, z1], color=color)

    # mincoords = np.array((self.minx, self.miny, self.minz))
    # maxcoords = np.array((self.maxx, self.maxy, self.maxz))
    # max_diff = max(maxcoords-mincoords)
    # ax.set_xlim(self.minx, self.minx+max_diff)
    # ax.set_ylim(self.miny, self.miny+max_diff)
    # ax.set_zlim(self.minz, self.minz+max_diff)
    plt.show()


def plot_tunnel_2d(tunnel: Tunnel, ax=None):
    if ax is None:
        ax = plt.gca()
    ds = np.arange(0, tunnel.distance, SPLINE_PLOT_PRECISSION)
    xs, ys = [], []
    for d in ds:
        p, d = tunnel.spline(d)
        x, y, z = p.flatten()
        xs.append(x)
        ys.append(y)
    color = np.array(list(np.random.uniform(0.2, 0.75, size=3)))
    ax.plot(xs, ys, c=color, linewidth=3)


def network_overview(tunnel_network: TunnelNetwork):
    print(f"Number of Nodes: {len(tunnel_network._nodes)}")
    print(f"Number of Edges: {len(tunnel_network._nodes)}")
    print(f"Number of Tunnels: {len(tunnel_network._tunnels)}")
    print(f"Number of Intersections: {len(tunnel_network._intersections)}")
