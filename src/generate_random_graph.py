import shapely
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import interpolate
import math

from helper_functions import *

from PARAMS import MAX_SEGMENT_INCLINATION, MIN_ANGLE_FOR_INTERSECTIONS, SPLINE_PLOT_PRECISSION


def add_noise_to_direction(direction,
                           horizontal_tendency,
                           horizontal_noise,
                           vertical_tendency,
                           vertical_noise):
    assert direction.size == 3
    th, ph = vector_to_angles(direction)
    horizontal_deviation = np.random.normal(
        horizontal_tendency, horizontal_noise)
    th = warp_angle_2pi(th + horizontal_deviation)
    ph = np.random.normal(vertical_tendency, vertical_noise)
    if abs(ph) > MAX_SEGMENT_INCLINATION:
        ph = MAX_SEGMENT_INCLINATION * ph/abs(ph)
    direction = angles_to_vector((th, ph))
    return direction


def correct_inclination(direction):
    assert direction.size == 3
    inclination = math.asin(direction[2])
    orientation = math.atan2(direction[1], direction[0])
    if abs(inclination) > MAX_SEGMENT_INCLINATION:
        z = math.sin(MAX_SEGMENT_INCLINATION) * inclination/abs(inclination)
        x = math.cos(MAX_SEGMENT_INCLINATION) * math.cos(orientation)
        y = math.cos(MAX_SEGMENT_INCLINATION) * math.sin(orientation)
        return np.array([x, y, z])
    else:
        return direction


def correct_direction_of_intersecting_tunnel(direction,
                                             intersecting_node,
                                             angle_threshold=MIN_ANGLE_FOR_INTERSECTIONS):
    if len(intersecting_node.connected_nodes) == 0:
        return direction
    th0, ph0 = vector_to_angles(direction)
    print("th0: {} // ph0: {}".format(np.rad2deg(th0), np.rad2deg(ph0)))
    closest_neg_angle, closest_pos_angle = None, None
    min_neg_difference, min_pos_difference = np.pi, np.pi
    for node in intersecting_node.connected_nodes:
        th1, ph1 = vector_to_angles(intersecting_node.xyz - node.xyz)
        difference = warp_angle_pi(th1-th0)
        if difference < 0 and abs(difference) < abs(min_neg_difference):
            min_neg_difference = difference
            closest_neg_angle = th1
        elif difference > 0 and abs(difference) < abs(min_pos_difference):
            min_pos_difference = difference
            closest_pos_angle = th1
    if abs(min_pos_difference) < angle_threshold and abs(min_neg_difference) < angle_threshold:
        return None
    if abs(min_neg_difference) < angle_threshold:
        thf = closest_neg_angle + angle_threshold
    elif abs(min_pos_difference) < angle_threshold:
        thf = closest_pos_angle - angle_threshold
    else:
        thf = th0
    final_direction = angles_to_vector((thf, ph0))
    print(f"thf: {np.rad2deg(thf)} // ph {np.rad2deg(ph0)}")
    print("#################################")
    return final_direction


def debug_plot(graph):
    assert isinstance(graph, Graph)
    plt.gca().clear()
    graph.plot2d()
    plt.draw()
    plt.waitforbuttonpress()


class Spline3D:
    """Wrapper around the scipy spline to 
    interpolate a series of 3d points along x,y and z"""
    def __init__(self, points):
        self.points = np.array(points)
        self.distances = [0 for _ in range(len(self.points))]
        for i in range(len(points)-1):
            self.distances[i+1] = self.distances[i] + \
                np.linalg.norm(points[i+1] - points[i])
        self.distance = self.distances[-1]
        degree = 3 if len(self.distances) > 3 else len(self.distances)-1
        self.xspline = interpolate.splrep(
            self.distances, self.points[:, 0], k=degree)
        self.yspline = interpolate.splrep(
            self.distances, self.points[:, 1], k=degree)
        self.zspline = interpolate.splrep(
            self.distances, self.points[:, 2], k=degree)

    def __call__(self, d):
        assert d >= 0 and d <= self.distance
        x = interpolate.splev(d, self.xspline)
        y = interpolate.splev(d, self.yspline)
        z = interpolate.splev(d, self.zspline)
        p = np.array([x, y, z])
        x1 = interpolate.splev(d+0.001, self.xspline)
        y1 = interpolate.splev(d+0.001, self.yspline)
        z1 = interpolate.splev(d+0.001, self.zspline)
        p1 = np.array([x1, y1, z1])
        v = p1 - p
        v /= np.linalg.norm(v)
        return p, v


class Node:
    __graph = None

    def __init__(self, coords=np.zeros(3), graph=None):
        if graph is None:
            assert isinstance(self.__graph, Graph)
            self.graph = self.__graph
        else:
            self.graph = graph
        self.connected_nodes = set()
        self.coords = coords
        self.tunnels = set()

    @classmethod
    def set_graph(cls, graph):
        cls.__graph = graph

    def add_tunnel(self, new_tunnel):
        """Nodes must keep track of what tunnels they are a part of
        this way, if a node is part of more than one tunnel, it means it
        is  an intersection"""
        if len(self.tunnels) == 0:
            self.tunnels.add(new_tunnel)
        if len(self.tunnels) == 1:
            if len(self.connected_nodes) == 2:
                list(self.tunnels)[0].split(self)

    def connect(self, node):
        """This function does everything to connect two nodes and update the 
        info in all relevant places"""
        assert isinstance(node, Node)
        self.graph.edges.append(Edge(self, node))

    def add_connection(self, node):
        """This function only inserts a new node in the connected nodes set"""
        self.connected_nodes.add(node)

    def remove_connection(self, node):
        """This function only deletes a node from the connected nodes set"""
        self.connected_nodes.remove(node)

    @property
    def xyz(self):
        return self.coords

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]


class Edge:
    def __init__(self, n0, n1):
        self.nodes = {n0, n1}
        self.subnodes = set()
        n0.add_connection(n1)
        n1.add_connection(n0)

    def __getitem__(self, index):
        return self.nodes[index]

    def plot2d(self, ax):
        nodes = list(self.nodes)
        x0 = nodes[0].x
        x1 = nodes[1].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        ax.plot([x0, x1], [y0, y1], c="k")

    def plot3d(self, ax):
        nodes = list(self.nodes)
        x0 = nodes[0].x
        x1 = nodes[1].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        z0 = nodes[0].z
        z1 = nodes[1].z
        ax.plot3D([x0, x1], [y0, y1], [z0, z1], c="k")


class TunnelParams:
    def __init__(self, params=None):
        self.__dict__ = {"distance": 100,
                         "starting_direction": (1, 0, 0),
                         "horizontal_tendency": 0,
                         "horizontal_noise": 0,
                         "vertical_tendency": 0,
                         "vertical_noise": 0,
                         "min_seg_length": 10,
                         "max_seg_length": 15}
        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self.__dict__[key] = params[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class Tunnel:
    def __init__(self, parent):
        assert isinstance(parent, Graph)
        self.parent = parent
        self.parent.tunnels.append(self)
        # The nodes should be ordered
        self.nodes = list()
        self._spline = None

    def split(self, node):
        assert node in self.nodes
        tunnel_1 = Tunnel(self.parent)
        tunnel_2 = Tunnel(self.parent)
        split_point = self.nodes.index(node)
        tunnel_1.set_nodes(self.nodes[:split_point+1])
        tunnel_2.set_nodes(self.nodes[split_point:])
        self.parent.remove_tunnel(self)

    def set_nodes(self, nodes):
        self.nodes = nodes
        self._spline = Spline3D([n.xyz for n in self.nodes])

    def add_node(self, node: Node):
        if len(self) != 0:
            self.nodes[-1].connect(node)
        node.add_tunnel(self)
        self.nodes.append(node)
        self.parent.add_node(node)
        self._spline = None

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([n.xyz for n in self.nodes])
        return self._spline

    def __len__(self):
        return len(self.nodes)

    @property
    def distance(self):
        return self.spline.distance

    def plot2d(self, ax):
        ds = np.arange(0, self.distance, SPLINE_PLOT_PRECISSION)
        xs, ys = [], []
        for d in ds:
            p, d = self.spline(d)
            x, y, z = p
            xs.append(x)
            ys.append(y)
        color = np.array(list(np.random.uniform(0.2, 0.75, size=3)))
        ax.plot(xs, ys, c=color, linewidth=3)


class Intersection:
    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.connected_tunnels = list()

    @property
    def n_tunnels(self):
        return len(self.connected_tunnels)


class Graph:
    def __init__(self):
        self.recalculate_control = True
        self.nodes = list()
        self.edges = list()
        self.tunnels = []  # List of lists
        self.intersections = []

    def add_node(self, node):
        self.recalculate_control = True
        if not node in self.nodes:
            self.nodes.append(node)

    def add_floating_tunnel(self, first_node_coords, tp: TunnelParams):
        previous_node = Node(first_node_coords)
        self.add_tunnel(previous_node, tp)

    def remove_tunnel(self, tunnel):
        assert tunnel in self.tunnels
        self.tunnels.remove(tunnel)

    def add_tunnel(self, first_node, tp: TunnelParams):
        tunnel = Tunnel(self)
        tunnel.add_node(first_node)
        previous_orientation = correct_direction_of_intersecting_tunnel(
            tp["starting_direction"], first_node)
        previous_node = first_node
        d = 0
        first_iteration = True
        while d < tp["distance"]:
            if not first_iteration:
                segment_orientation = add_noise_to_direction(
                    previous_orientation, tp["horizontal_tendency"], tp["horizontal_noise"], tp["vertical_tendency"], tp["vertical_noise"])
            else:
                segment_orientation = previous_orientation
            segment_length = np.random.uniform(
                tp["min_seg_length"], tp["max_seg_length"])
            d += segment_length
            new_node_coords = previous_node.xyz + segment_orientation * segment_length
            new_node = Node(coords=new_node_coords)
            tunnel.add_node(new_node)
            previous_node.connect(new_node)
            previous_node = new_node
            previous_orientation = segment_orientation
            first_iteration = False

    @property
    def minx(self):
        return min([n.x for n in self.nodes])

    @property
    def miny(self):
        return min([n.y for n in self.nodes])

    @property
    def minz(self):
        return min([n.z for n in self.nodes])

    @property
    def maxx(self):
        return max([n.x for n in self.nodes])

    @property
    def maxy(self):
        return max([n.y for n in self.nodes])

    @property
    def maxz(self):
        return max([n.z for n in self.nodes])

    def connect_with_tunnel(self, n1, n2):
        pass

    def plot2d(self, ax=None):
        if ax is None:
            ax = plt.gca()

        for edge in self.edges:
            edge.plot2d(ax)
        for node in self.nodes:
            ax.scatter(node.x, node.y, c="b")
        for tunnel in self.tunnels:
            tunnel.plot2d(ax)
        mincoords = np.array((self.minx, self.miny))
        maxcoords = np.array((self.maxx, self.maxy))
        max_diff = max(maxcoords-mincoords)
        ax.set_xlim(min(mincoords), max(maxcoords))
        ax.set_ylim(min(mincoords), max(maxcoords))
        if ax is None:
            plt.show()

    def plot3d(self):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
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

        #mincoords = np.array((self.minx, self.miny, self.minz))
        #maxcoords = np.array((self.maxx, self.maxy, self.maxz))
        #max_diff = max(maxcoords-mincoords)
        #ax.set_xlim(self.minx, self.minx+max_diff)
        #ax.set_ylim(self.miny, self.miny+max_diff)
        #ax.set_zlim(self.minz, self.minz+max_diff)
        plt.show()


def main():
    n_rows = 5
    n_cols = 5
    fig = plt.figure(figsize=(8, 8))
    axis = plt.subplot(1, 1, 1)
    plt.show(block=False)
    while True:
        tunnel_params = TunnelParams({"distance": 100,
                                      "starting_direction": np.array((1, 0, 0)),
                                      "horizontal_tendency": np.deg2rad(0),
                                      "horizontal_noise": np.deg2rad(20),
                                      "vertical_tendency": np.deg2rad(10),
                                      "vertical_noise": np.deg2rad(5),
                                      "min_seg_length": 20,
                                      "max_seg_length": 30})
        graph = Graph()
        Node.set_graph(graph)  # This is so all nodes share the same graph
        graph.add_floating_tunnel(np.array((0, 0, 0)), tunnel_params)
        debug_plot(graph)
        node = graph.nodes[-3]
        graph.add_tunnel(node, tunnel_params)
        debug_plot(graph)
        tunnel_params["starting_direction"] = np.array((0, 1, 0))
        graph.add_tunnel(node, tunnel_params)
        debug_plot(graph)


if __name__ == "__main__":
    main()
