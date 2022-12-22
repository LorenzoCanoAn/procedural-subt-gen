"""This file contains all the data structures that interpret the graph as a set of tunnels
and build it as such"""
from subt_proc_gen.graph import Graph, Node
from subt_proc_gen.PARAMS import *
import numpy as np
from subt_proc_gen.helper_functions import (
    vector_to_angles,
    warp_angle_2pi,
    warp_angle_pi,
    angles_to_vector,
    angle_between_angles,
)
import math
from scipy import interpolate


def add_noise_to_direction(
    direction, horizontal_tendency, horizontal_noise, vertical_tendency, vertical_noise
):
    assert direction.size == 3
    th, ph = vector_to_angles(direction)
    horizontal_deviation = np.random.normal(horizontal_tendency, horizontal_noise)
    th = warp_angle_2pi(th + horizontal_deviation)
    ph = np.random.normal(vertical_tendency, vertical_noise)
    if abs(ph) > MAX_SEGMENT_INCLINATION:
        ph = MAX_SEGMENT_INCLINATION * ph / abs(ph)
    direction = angles_to_vector((th, ph))
    return direction


def correct_direction_of_intersecting_tunnel(
    i_dir_vect, intersection_node, angle_threshold=MIN_ANGLE_FOR_INTERSECTIONS
):
    if len(intersection_node.connected_nodes) == 0:
        return i_dir_vect

    th0, ph0 = vector_to_angles(i_dir_vect)
    closest_neg_angle, closest_pos_angle = None, None
    min_neg_difference, min_pos_difference = np.pi, np.pi

    for n_node, node in enumerate(intersection_node.connected_nodes):
        th1, ph1 = vector_to_angles(node.xyz - intersection_node.xyz)
        if angle_between_angles(th1, th0) < np.deg2rad(3):
            return None
        difference = warp_angle_pi(th1 - th0)

        if difference < 0 and abs(difference) < abs(min_neg_difference):
            min_neg_difference = difference
            closest_neg_angle = th1
        elif difference > 0 and abs(difference) < abs(min_pos_difference):
            min_pos_difference = difference
            closest_pos_angle = th1
    if angle_between_angles(closest_neg_angle, closest_pos_angle) < angle_threshold * 2:
        return None
    if abs(min_neg_difference) < angle_threshold:
        thf = closest_neg_angle + angle_threshold
    elif abs(min_pos_difference) < angle_threshold:
        thf = closest_pos_angle - angle_threshold
    else:
        thf = th0

    final_direction = angles_to_vector((thf, ph0))
    return final_direction


class Spline3D:
    """Wrapper around the scipy spline to
    interpolate a series of 3d points along x,y and z"""

    def __init__(self, points):
        self.points = np.array(points)
        self.distances = [0 for _ in range(len(self.points))]
        for i in range(len(points) - 1):
            self.distances[i + 1] = self.distances[i] + np.linalg.norm(
                points[i + 1] - points[i]
            )
        self.length = self.distances[-1]
        degree = 3 if len(self.distances) > 3 else len(self.distances) - 1
        self.xspline = interpolate.splrep(self.distances, self.points[:, 0], k=degree)
        self.yspline = interpolate.splrep(self.distances, self.points[:, 1], k=degree)
        self.zspline = interpolate.splrep(self.distances, self.points[:, 2], k=degree)

    def __call__(self, d):
        assert d >= 0 and d <= self.length
        x = interpolate.splev(d, self.xspline)
        y = interpolate.splev(d, self.yspline)
        z = interpolate.splev(d, self.zspline)
        p = np.array([x, y, z], ndmin=2)
        x1 = interpolate.splev(d + 0.001, self.xspline)
        y1 = interpolate.splev(d + 0.001, self.yspline)
        z1 = interpolate.splev(d + 0.001, self.zspline)
        p1 = np.array([x1, y1, z1], ndmin=2)
        v = p1 - p
        v /= np.linalg.norm(v)
        return p, v


class TunnelParams(dict):
    def __init__(self, params=None, random=False):
        super().__init__()
        if random:
            self.random()
        else:
            self["distance"] = 100
            self["starting_direction"] = (1, 0, 0)
            self["horizontal_tendency"] = 0
            self["horizontal_noise"] = 0
            self["vertical_tendency"] = 0
            self["vertical_noise"] = 0
            self["min_seg_length"] = 10
            self["max_seg_length"] = 15

        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self[key] = params[key]

    def random(self):
        self["distance"] = np.random.uniform(20, 200)
        th = np.deg2rad(np.random.uniform(-180, 180))
        ph = np.deg2rad(np.random.uniform(-20, 20))
        self["starting_direction"] = angles_to_vector((th, ph))
        self["horizontal_tendency"] = np.deg2rad(np.random.uniform(-10, 10))
        self["horizontal_noise"] = np.deg2rad(np.random.uniform(0, 5))
        self["vertical_tendency"] = np.deg2rad(np.random.uniform(-5, 5))
        self["vertical_noise"] = np.deg2rad(np.random.uniform(0, 2))
        self["min_seg_length"] = self["distance"] * np.random.uniform(0.1, 0.2)
        self["max_seg_length"] = self["distance"] * np.random.uniform(0.25, 0.4)


class Tunnel:
    def __init__(self, parent, seed, params=TunnelParams()):
        """A tunnel can be started from three different seeds:
        - If the seed is a CaveNode: The tunnel grows from said node according to its parameters
        - If the seed is an np.ndarray representing a point, a CaveNode is created in that position, and a tunnel is grown from it
        - If teh seed is a list of CaveNodes, they are set as the Tunnel nodes
        """
        assert isinstance(parent, TunnelNetwork)
        self._parent = parent
        self._params = params

        self._parent.add_tunnel(self)

        # Internal variables that should be accessed from functions
        self._nodes = list()
        self._spline = None
        successful_growth = True
        if isinstance(seed, CaveNode):
            self.add_node(seed)
            successful_growth = self.grow_tunnel()
        elif isinstance(seed, np.ndarray) and len(seed) == 3:
            self.add_node(CaveNode(seed))
            successful_growth = self.grow_tunnel()
        elif isinstance(seed, list) or isinstance(seed, set):
            self.set_nodes(seed)

        if not successful_growth:
            self.delete()

    def delete(self):
        """
        When a tunnel is deleted, all it's exclusive nodes should be deleted, and it should
        delete itself fromm all the other nodes and from it's parent
        """
        try:
            for node in self._nodes:
                assert isinstance(node, CaveNode)
                node.remove_tunnel(self)
                if len(node.tunnels) == 0:
                    node.delete()
            self._parent.remove_tunnel(self)
        except:
            exit()

    def set_nodes(self, nodes):
        self._nodes = nodes
        for node in self._nodes:
            assert isinstance(node, CaveNode)
            node.tunnels.add(self)
        self._spline = Spline3D([n.xyz for n in self._nodes])

    def add_node(self, node):
        assert isinstance(node, CaveNode)
        self._nodes.append(node)
        node.add_tunnel(self)
        self._spline = None

    def __getitem__(self, index):
        return self._nodes[index]

    def grow_tunnel(self):
        """This function is called after setting the first node of the tunnel"""
        tp = self._params  # for readability

        previous_orientation = correct_direction_of_intersecting_tunnel(
            self._params["starting_direction"], self[0]
        )
        if previous_orientation is None:
            return False

        d = 0
        n = 1
        assert len(self) == 1
        previous_node = self[0]
        while d < tp["distance"]:
            if not n == 1:
                segment_orientation = add_noise_to_direction(
                    previous_orientation,
                    tp["horizontal_tendency"],
                    tp["horizontal_noise"],
                    tp["vertical_tendency"],
                    tp["vertical_noise"],
                )
            else:
                segment_orientation = previous_orientation
            segment_length = np.random.uniform(
                tp["min_seg_length"], tp["max_seg_length"]
            )
            d += segment_length
            new_node_coords = self[n - 1].xyz + segment_orientation * segment_length
            new_node = CaveNode(coords=new_node_coords)
            self.add_node(new_node)
            new_node.connect(previous_node)
            previous_orientation = segment_orientation
            previous_node = new_node
            n += 1
        return True

    def common_nodes(self, tunnel):
        assert isinstance(tunnel, Tunnel)
        common_nodes = set()
        for node in self.nodes:
            if node in tunnel.nodes:
                common_nodes.add(node)
        return common_nodes

    @property
    def nodes(self):
        return self._nodes

    @property
    def distance(self):
        return self.spline.length

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([n.xyz for n in self._nodes])
        return self._spline

    @property
    def end_nodes(self):
        return (self._nodes[0], self._nodes[-1])

    def __len__(self):
        return len(self._nodes)


class CaveNode(Node):
    def __init__(self, coords=np.array((0, 0, 0))):
        super().__init__(coords)
        self._tunnels = set()

    @property
    def tunnels(self):
        return self._tunnels

    def delete(self):
        """To delete a node, it it necessary to remove it from its graph and connections"""
        self._graph.delete_node(self)

    def add_tunnel(self, new_tunnel):
        """Nodes must keep track of what tunnels they are a part of
        this way, if a node is part of more than one tunnel, it means it
        is  an intersection"""
        self._tunnels.add(new_tunnel)

    def remove_tunnel(self, tunnel):
        assert isinstance(tunnel, Tunnel)
        if tunnel not in self._tunnels:
            raise Exception(
                "Trying to remove a tunnel from this node that this node is not a part of"
            )
        else:
            self._tunnels.remove(tunnel)


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = set()
        self._intersections = set()

    def remove_tunnel(self, tunnel):
        assert tunnel in self._tunnels, Exception(
            "Trying to remove a tunnel that is not in the tunnel network"
        )
        self._tunnels.remove(tunnel)

    def add_tunnel(self, tunnel: Tunnel):
        self._tunnels.add(tunnel)

    @property
    def tunnels(self):
        return self._tunnels

    @property
    def intersections(self):
        # Return all nodes that have more than one tunnel assigned
        self._intersections.clear()
        for node in self._nodes:
            assert isinstance(node, CaveNode), TypeError(
                f"node is of type: {type(node)}"
            )
            if len(node.tunnels) > 2:
                self._intersections.add(node)
        return self._intersections
