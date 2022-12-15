"""This file contains all the data structures that interpret the graph as a set of tunnels
and build it as such"""
from graph import Graph, Node
from PARAMS import *
import numpy as np
from helper_functions import vector_to_angles, warp_angle_2pi, warp_angle_pi, angles_to_vector
import math
from scipy import interpolate


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


class TunnelParams(dict):
    def __init__(self, params=None):
        super().__init__()
        self["distance"] = 100,
        self["starting_direction"] = (1, 0, 0),
        self["horizontal_tendency"] = 0,
        self["horizontal_noise"] = 0,
        self["vertical_tendency"] = 0,
        self["vertical_noise"] = 0,
        self["min_seg_length"] = 10,
        self["max_seg_length"] = 15

        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self[key] = params[key]


class Tunnel:
    def __init__(self, parent, seed, params = TunnelParams()):
        assert isinstance(parent, TunnelNetwork)
        self.parent = parent
        self.params = params

        self.parent.add_tunnel(self)

        # Internal variables that should be accessed from functions
        self._nodes = list()
        self._spline = None

        if isinstance(seed, Node):
            self.add_node(seed)
            self.grow_tunnel()
        elif isinstance(seed, np.ndarray) and len(seed) == 3:
            self.add_node(Node(seed))
            self.grow_tunnel()
        elif isinstance(seed, list) or isinstance(seed, set):
            self.set_nodes(seed)

    def split(self, node):
        assert node in self._nodes
        split_point = self._nodes.index(node)
        for node in self._nodes:
            node.tunnels.remove(self)
        tunnel_1 = Tunnel(self.parent,self._nodes[:split_point+1])
        tunnel_2 = Tunnel(self.parent,self._nodes[split_point:])
        self.parent.remove_tunnel(self)

    def set_nodes(self, nodes):
        self._nodes = nodes
        for node in self._nodes:
            assert isinstance(node, Node)
            node.tunnels.add(self)
        self._spline = Spline3D([n.xyz for n in self._nodes])

    def add_node(self, node: Node):
        if len(self) != 0:
            self._nodes[-1].connect(node)
        self._nodes.append(node)
        node.add_tunnel(self)
        self.parent.add_node(node)
        self._spline = None

    def __getitem__(self, index):
        return self._nodes[index]

    def grow_tunnel(self):
        """This function is called after setting the first node of the tunnel"""
        tp = self.params  # for readability
        previous_orientation = correct_direction_of_intersecting_tunnel(
            self.params["starting_direction"], self[0])

        d = 0
        n = 1
        while d < tp["distance"]:
            if not n == 1:
                segment_orientation = add_noise_to_direction(
                    previous_orientation, tp["horizontal_tendency"], tp["horizontal_noise"], tp["vertical_tendency"], tp["vertical_noise"])
            else:
                segment_orientation = previous_orientation
            segment_length = np.random.uniform(
                tp["min_seg_length"], tp["max_seg_length"])
            d += segment_length
            new_node_coords = self[n-1].xyz + \
                segment_orientation * segment_length
            new_node = Node(coords=new_node_coords)
            self.add_node(new_node)
            previous_orientation = segment_orientation
            n+=1

    @property
    def distance(self):
        return self.spline.distance

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([n.xyz for n in self._nodes])
        return self._spline

    def __len__(self):
        return len(self._nodes)


class Intersection:
    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.connected_tunnels = list()

    @property
    def n_tunnels(self):
        return len(self.connected_tunnels)


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = set()
        self._intersections = set()

    def remove_tunnel(self, tunnel):
        assert tunnel in self._tunnels
        self._tunnels.remove(tunnel)

    def add_tunnel(self, tunnel: Tunnel):
        self._tunnels.add(tunnel)
