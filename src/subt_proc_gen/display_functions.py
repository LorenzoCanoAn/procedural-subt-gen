"""Functions to display and debug the graph generation process"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from subt_proc_gen.PARAMS import *
import pyvista as pv
from subt_proc_gen.tunnel import Tunnel, Graph, TunnelNetwork
from subt_proc_gen.graph import Node, Edge
import vtk


def plot_node(plotter: pv.Plotter, node: Node, radius=0.3, color="b"):
    return plotter.add_mesh(pv.Sphere(radius=radius, center=node.xyz), color=color)


def plot_edge(plotter: pv.Plotter, edge: Edge, radius=0.1, color="b"):
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
    radius=0.3,
    color="b",
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
    radius=0.1,
    color="b",
):
    actors = []
    for edge in edges:
        actors.append(plot_edge(plotter, edge, radius=radius, color=color))
