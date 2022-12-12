"""This file contains all the data structures that interpret the graph as a set of tunnels
and build it as such"""
from graph import Graph, Node
from PARAMS import *
import numpy as np
from helper_functions import vector_to_angles, warp_angle_2pi, warp_angle_pi, angles_to_vector
import math
from scipy import interpolate

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
    
    @classmethod
    def from_initial_coordinates(cls, graph, first_node_coords, tp: TunnelParams):
        tunnel = Tunnel(graph)

        previous_node = Node(first_node_coords)
        graph.add_tunnel(previous_node, tp)

    @classmethod
    def from_node(self, first_node, tp: TunnelParams):
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