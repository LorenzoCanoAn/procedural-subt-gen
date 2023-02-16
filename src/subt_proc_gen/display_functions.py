"""Functions to display and debug the graph generation process"""
import matplotlib.pyplot as plt
import numpy as np
from subt_proc_gen.PARAMS import *
import pyvista as pv
from subt_proc_gen.tunnel import Tunnel, Graph, TunnelNetwork
from dataclasses import dataclass


@dataclass
class graph_plotting_params:
    intersection_color: str = "b"
    # Grown tunnel parms
    gt_nc: str = "b"
    gt_na: float = 1
    gt_sc: str = "b"
    gt_sa: float = 1
    gt_ec: str = "b"
    gt_ea: float = 1
    # Tunnel between nodes params
    tbn_nc: str = "r"
    tbn_na: float = 1
    tbn_sc: str = "r"
    tbn_sa: float = 1
    tbn_ec: str = "r"
    tbn_ea: float = 1


def debug_plot(graph, in_3d=False, wait="", clear=False):
    if clear:
        plt.gca().clear()
    if in_3d:
        ax = plt.gcf().add_subplot(projection="3d")
        plot_graph_3d(graph)
    else:
        plot_graph_2d(graph)
    if wait == "click":
        plt.draw()
        plt.waitforbuttonpress()
    elif wait == "close":
        plt.show()
    else:
        plt.draw()


def plot_graph_2d(graph, ax=None):
    if ax is None:
        ax = plt.gca()

    for edge in graph.edges:
        edge.plot2d(ax)
    for node in graph.nodes:
        ax.scatter(node.x, node.y, c="b", s=300)
    for tunnel in graph._tunnels:
        plot_spline(tunnel.spline, ax)
    mincoords = np.array((graph.minx, graph.miny))
    maxcoords = np.array((graph.maxx, graph.maxy))
    max_diff = max(maxcoords - mincoords)
    ax.set_xlim(-20, 280)
    ax.set_ylim(-150, 150)


def plot_graph_3d(graph: TunnelNetwork, ax=None):
    if ax is None:
        ax = plt.gca()
    for n_node, node in enumerate(graph.nodes):
        ax.scatter3D(node.x, node.y, node.z, c="b", s=300)
        ax.plot3D([node.x, node.x], [node.y, node.y], [node.z, 0], c="g", linewidth=3)
        ax.scatter3D(node.x, node.y, 0, c="g", s=150)
    for tunnel in graph.tunnels:
        assert isinstance(tunnel, Tunnel)
        # Plot initial node
        if tunnel.tunnel_type == "grown":
            ax.scatter3D(
                tunnel.nodes[0].x, tunnel.nodes[0].y, tunnel.nodes[0].z, c="r", s=300
            )
        elif tunnel.tunnel_type == "between_nodes":
            ax.scatter3D(
                tunnel.nodes[0].x, tunnel.nodes[0].y, tunnel.nodes[0].z, c="k", s=300
            )
            ax.scatter3D(
                tunnel.nodes[-1].x, tunnel.nodes[-1].y, tunnel.nodes[-1].z, c="k", s=300
            )
        # Plot edges
        for n_node in range(1, len(tunnel.nodes)):
            n0 = tunnel.nodes[n_node - 1]
            n1 = tunnel.nodes[n_node]
            if tunnel.tunnel_type == "grown":
                color = "k"
            elif tunnel.tunnel_type == "between_nodes":
                color = "r"
            ax.plot3D(
                [n0.x, n1.x], [n0.y, n1.y], [n0.z, n1.z], c=color, linewidth=3, alpha=1
            )
            ax.plot3D(
                [n0.x, n1.x], [n0.y, n1.y], [0, 0], c=color, linewidth=3, alpha=0.3
            )
    mincoords = np.min(np.array((graph.minx, graph.miny, graph.minz)))
    maxcoords = np.max(np.array((graph.maxx, graph.maxy, graph.maxz)))
    xx, yy = np.meshgrid(
        range(int(graph.minx - 10), int(graph.maxx + 10), 5),
        range(int(graph.miny - 10), int(graph.maxy + 10), 5),
    )
    z = xx * 0
    ax.plot_surface(xx, yy, z, alpha=0.3)
    ax.set_xticks([0, 70, 140])
    ax.set_xlim(-10, 150)
    ax.set_yticks([-70, 0, 70])
    ax.set_ylim(-80, 80)
    ax.set_zticks([-30, 0, 30])
    ax.set_zlim(-40, 40)
    ax.set_xlabel("X", fontsize=30, rotation=0, labelpad=15)
    ax.set_ylabel("Y", fontsize=30, rotation=0, labelpad=15)
    ax.set_zlabel("Z", fontsize=30, rotation=0, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontsize(20)


def plot_spline(spline, ax=None, color="b"):
    if ax is None:
        ax = plt.gca()
    ds = np.arange(0, spline.length, SPLINE_PLOT_PRECISSION)
    xs, ys = [], []
    for d in ds:
        p, d = spline(d)
        x, y, z = p.flatten()
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, c=color, linewidth=3, alpha=0.5)


def network_overview(tunnel_network):
    print(f"Number of Nodes: {len(tunnel_network._nodes)}")
    print(f"Number of Edges: {len(tunnel_network._nodes)}")
    print(f"Number of Tunnels: {len(tunnel_network._tunnels)}")
    print(f"Number of Intersections: {len(tunnel_network._intersections)}")
