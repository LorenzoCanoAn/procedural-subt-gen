"""Functions to display and debug the graph generation process"""
import numpy as np
import pyvista as pv
from subt_proc_gen.tunnel import Tunnel, Graph, TunnelNetwork
from subt_proc_gen.graph import Node, Edge
from subt_proc_gen.geometry import Spline3D
from subt_proc_gen.mesh_generation import TunnelNewtorkMeshGenerator


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


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
    if radius is None:
        radius = 0.3
    if color is None:
        color = "b"
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
    return actors


def plot_spline(
    plotter: pv.Plotter,
    spline: Spline3D,
    radius=None,
    color=None,
):
    if radius is None:
        radius = 0.1
    if color is None:
        color = "r"
    line = lines_from_points(points=spline.discretize(0.5)[1])
    tube = line.tube(radius=radius)
    return plotter.add_mesh(tube, color=color)


def plot_splines(
    plotter: pv.Plotter,
    tunnel_network: TunnelNetwork,
    radius=None,
    color=None,
):
    for tunnel in tunnel_network.tunnels:
        plot_spline(plotter=plotter, spline=tunnel.spline, radius=radius, color=color)


def plot_ptcl(
    plotter: pv.Plotter,
    points: np.ndarray,
    radius=None,
    color=None,
):
    points = np.reshape(points, [-1, 3])
    if radius is None:
        radius = 0.01
    if color is None:
        color = "m"
    mesh = pv.PolyData(points)
    glyphs = mesh.glyph(
        orient=False,
        geom=pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=20),
        scale=False,
    )
    return plotter.add_mesh(glyphs, color=color)


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
    return [node_actors] + edge_actors


def plot_tunnel_ptcls(
    plotter: pv.Plotter,
    mesh_generator: TunnelNewtorkMeshGenerator,
    size=None,
    color=None,
):
    if color is None:
        color = "m"
    actors = []
    for tunnel in mesh_generator._tunnel_network.tunnels:
        actors.append(
            plot_ptcl(
                plotter,
                mesh_generator._ptcl_of_tunnels[tunnel],
                radius=size,
                color=color,
            )
        )
    return actors


def plot_intersection_ptcls(
    plotter: pv.Plotter,
    mesh_generator: TunnelNewtorkMeshGenerator,
    size=None,
    color=None,
):
    if color is None:
        color = "c"
    actors = []
    for intersection in mesh_generator._tunnel_network.intersections:
        actors.append(
            plot_ptcl(
                plotter,
                mesh_generator.ptcl_of_intersections(intersection),
                radius=size,
                color=color,
            )
        )
    return actors
