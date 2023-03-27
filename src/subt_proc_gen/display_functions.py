"""Functions to display and debug the graph generation process"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from subt_proc_gen.PARAMS import *
import pyvista as pv
from subt_proc_gen.tunnel import Tunnel, Graph, TunnelNetwork
from subt_proc_gen.graph import Node, Edge
import vtk


def plot_node(plotter: pv.Plotter, node: Node, radius=None, color=None):
    if radius is None:
        radius = 0.3
    if color is None:
        color = "b"
    return plotter.add_mesh(pv.Sphere(radius=radius, center=node.xyz), color=color)


def plot_edge(plotter: pv.Plotter, edge: Edge, radius=None, color=None):
    if radius is None:
        radius = 0.1
    if color is None:
        color = "b"
    ni, nj = edge.nodes
    return plotter.add_mesh(
        pv.Tube(pointa=ni.xyz, pointb=nj.xyz, radius=radius), color=color
    )


def plot_xyz_axis(plotter: pv.Plotter):
    pxa, pxb = (0, 0, 0), (50, 0, 0)
    pya, pyb = (0, 0, 0), (0, 50, 0)
    pza, pzb = (0, 0, 0), (0, 0, 50)
    actors = []
    actors.append(
        plotter.add_mesh(pv.Tube(pointa=pxa, pointb=pxb, radius=0.5), color="r")
    )
    actors.append(
        plotter.add_mesh(pv.Tube(pointa=pya, pointb=pyb, radius=0.5), color="g")
    )
    actors.append(
        plotter.add_mesh(pv.Tube(pointa=pza, pointb=pzb, radius=0.5), color="b")
    )
    return actors


def plot_nodes(
    plotter: pv.Plotter,
    nodes: list[Node] | set[Node] | tuple[Node],
    radius=None,
    color=None,
):
    n_nodes = len(nodes)
    point_array = np.zeros([n_nodes, 3])
    for i, node in enumerate(nodes):
        point_array[i] = node.xyz
    mesh = pv.PolyData(point_array)
    glyphs = mesh.glyph(
        orient=False,
        geom=pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20),
        scale=False,
    )
    return plotter.add_mesh(glyphs, color=color)


def plot_edges(
    plotter: pv.Plotter,
    edges: list[Edge] | set[Edge] | tuple[Edge],
    radius=None,
    color=None,
):
    actors = []
    for edge in edges:
        actors.append(plot_edge(plotter, edge, radius=radius, color=color))


def plot_graph(
    plotter: pv.Plotter,
    graph: Graph,
    node_rad=None,
    edge_rad=None,
    node_color=None,
    edge_color=None,
):
    node_actors = plot_nodes(plotter, graph.nodes, radius=node_rad, color=node_color)
    edge_actors = plot_edges(plotter, graph.edges, radius=edge_rad, color=edge_color)
    return node_actors + edge_actors
