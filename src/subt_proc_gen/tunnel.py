"""This file contains all the data structures that interpret the graph as a set of tunnels
and build it as such"""
from subt_proc_gen.graph import Graph, Node
from subt_proc_gen.PARAMS import *
from subt_proc_gen.spline import Spline3D
import numpy as np
import matplotlib.pyplot as plt
from subt_proc_gen.helper_functions import (
    vector_to_angles,
    warp_angle_2pi,
    warp_angle_pi,
    any_point_close,
    angles_to_vector,
    angle_between_angles,
)
import math


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
        # Angular distance to this connection
        difference = warp_angle_pi(th1 - th0)
        # Is this angular distance the new closest one?
        if difference < 0 and abs(difference) < abs(min_neg_difference):
            min_neg_difference = difference
            closest_neg_angle = th1
        elif difference > 0 and abs(difference) < abs(min_pos_difference):
            min_pos_difference = difference
            closest_pos_angle = th1
    # Check if there is enough space between the two closest connections
    if not closest_neg_angle is None and not closest_pos_angle is None:
        if (
            angle_between_angles(closest_neg_angle, closest_pos_angle)
            < angle_threshold * 2
        ):
            return None
    if abs(min_neg_difference) < angle_threshold:
        thf = closest_neg_angle + angle_threshold
    elif abs(min_pos_difference) < angle_threshold:
        thf = closest_pos_angle - angle_threshold
    else:
        thf = th0
    final_direction = angles_to_vector((thf, ph0))
    return final_direction


def check_insertion_angle_of_candidate_node_for_intersection(
    candidate_node, intersection_node, angle_threshold=MIN_ANGLE_FOR_INTERSECTIONS
):
    assert isinstance(candidate_node, CaveNode)
    assert isinstance(intersection_node, CaveNode)
    if len(intersection_node._connected_nodes) == 0:
        return True
    i_dir_vect = candidate_node.xyz - intersection_node.xyz
    th0, ph0 = vector_to_angles(i_dir_vect)
    for n_node, node in enumerate(intersection_node._connected_nodes):
        if node is candidate_node:
            continue
        th1, ph1 = vector_to_angles(node.xyz - intersection_node.xyz)
        th_0_1 = angle_between_angles(th0, th1)
        if abs(th_0_1) < angle_threshold:
            return False
    return True


class TunnelParams(dict):
    def __init__(self, params=None, random=False):
        super().__init__()
        if random:
            self.random()
        else:
            self["distance"] = 100
            self["starting_direction"] = np.array((1, 0, 0))
            self["horizontal_tendency"] = 0
            self["horizontal_noise"] = 0
            self["vertical_tendency"] = 0
            self["vertical_noise"] = 0
            self["segment_length"] = 10
            self["segment_length_noise"] = 15
            # Params for a tunnel between two nodes
            self["node_position_noise"] = 5

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
        self["segment_length"] = self["distance"] * np.random.uniform(0.1, 0.2)
        self["segment_length_noise"] = self["segment_length"] / 5
        self["node_position_noise"] = np.random.uniform(5, 7)


class CaveNode(Node):
    def __init__(self, coords=np.array((0, 0, 0))):
        super().__init__(coords)
        self._tunnels = list()

    @property
    def tunnels(self):
        return self._tunnels

    def add_tunnel(self, new_tunnel):
        """Nodes must keep track of what tunnels they are a part of
        this way, if a node is part of more than one tunnel, it means it
        is  an intersection"""
        self._tunnels.append(new_tunnel)

    def remove_tunnel(self, tunnel):
        assert isinstance(tunnel, Tunnel)
        if tunnel not in self._tunnels:
            raise Exception(
                "Trying to remove a tunnel from this node that this node is not a part of"
            )
        else:
            self._tunnels.remove(tunnel)


class Tunnel:
    def __init__(
        self,
        parent,
        initial_node=None,
        final_node=None,
        params=TunnelParams(),
        override_checks=False,
    ):
        """A tunnel can be started from three different seeds:
        - If the seed is a CaveNode: The tunnel grows from said node according to its parameters
        - If the seed is an np.ndarray representing a point, a CaveNode is created in that position, and a tunnel is grown from it
        - If teh seed is a list of CaveNodes, they are set as the Tunnel nodes
        """
        assert isinstance(parent, TunnelNetwork)
        self._parent = parent
        self._params = params

        self._parent.add_tunnel(self)
        self.tunnel_type = None
        # Internal variables that should be accessed from functions
        self._nodes = list()
        self._spline = None
        successful_growth = True
        reson = None
        if isinstance(initial_node, CaveNode):
            if not final_node is None:
                self.tunnel_type = "between_nodes"
                successful_growth = self.grow_between_nodes(initial_node, final_node)
            else:
                self.add_node(initial_node)
                self.tunnel_type = "grown"
                successful_growth = self.grow_tunnel()
        print(f"Successful_grouth: {successful_growth}")
        print(f"nodes: {self.nodes}")
        if not successful_growth and not override_checks:
            # print("Tunnel generation not successful, reason:")
            # print(f"\t {reason}")
            self.delete()
            self.success = False
        else:
            self.success = True

    def delete(self):
        """
        When a tunnel is deleted, all it's exclusive nodes should be deleted, and it should
        delete itself fromm all the other nodes and from it's parent
        """
        if len(self._nodes) == 2:
            print("Deleting a 2 ndoe tunnel")
            self._nodes[0].remove_connection(self._nodes[1])
            self._nodes[1].remove_connection(self._nodes[0])
        for node in self._nodes:
            assert isinstance(node, CaveNode)
            node.remove_tunnel(self)
            if len(node.tunnels) == 0:
                node.delete()
        self._parent.remove_tunnel(self)

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

    def __getitem__(self, index: int) -> CaveNode:
        return self._nodes[index]

    def __len__(self):
        return len(self._nodes)

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
                tp["segment_length"] - tp["segment_length_noise"],
                tp["segment_length"] + tp["segment_length_noise"],
            )
            d += segment_length
            new_node_coords = self[n - 1].xyz + segment_orientation * segment_length
            new_node = CaveNode(coords=new_node_coords)
            self.add_node(new_node)
            new_node.connect(previous_node)
            previous_orientation = segment_orientation
            previous_node = new_node
            n += 1
        print(f"Inside function nodes: {self.nodes}")
        if len(self.nodes) < 2:
            print("Unsuccesful tunnel: Only one node")
            return False
        if not check_insertion_angle_of_candidate_node_for_intersection(
            self[1], self[0]
        ):
            print("Unsuccesful tunnel: Angle with intersection")
            return False
        if not self.check_collissions():
            print("Unsuccessful tunnel: Collides")
            return False
        return True

    def grow_between_nodes(self, initial_node, final_node):
        assert isinstance(initial_node, CaveNode)
        assert isinstance(final_node, CaveNode)
        diff = final_node.coords - initial_node.coords
        dist = np.linalg.norm(diff)
        vect = diff / dist
        n = int(np.ceil(dist / self._params["segment_length"]))
        segment_length = dist / n
        ds = [i * segment_length for i in range(1, n)]
        D = 0

        self.add_node(initial_node)
        previous_node = initial_node
        for d in ds:
            new_node_coords = np.reshape(
                vect * d
                + initial_node.coords
                + np.random.uniform(0, self._params["node_position_noise"], (1, 3)),
                (3,),
            )
            new_node = CaveNode(new_node_coords)
            self.add_node(new_node)
            new_node.connect(previous_node)
            previous_node = new_node
        self.add_node(final_node)
        final_node.connect(previous_node)
        # Now i check that the inclinations of the segments are reasonable
        for i in range(1, len(self)):
            n0 = self[i - 1]
            n1 = self[i]
            vect = n1.coords - n0.coords
            th, ph = vector_to_angles(vect)
            if abs(ph) > MAX_SEGMENT_INCLINATION:
                print("Unsuccessful tunnel: Segment with too much inclination")
                return False
        if not check_insertion_angle_of_candidate_node_for_intersection(
            self[1], self[0]
        ):
            print("Unsuccessfull grouth: Intersection angle error with first node")
            return False
        if not check_insertion_angle_of_candidate_node_for_intersection(
            self[-2], self[-1]
        ):
            print("Unsuccessfull grouth: Intersection angle error with second node")
            return False
        if not self.check_collissions():
            print("Unsuccessfull grouth: Collision check failed")
            return False
        return True

    def common_nodes(self, tunnel):
        assert isinstance(tunnel, Tunnel)
        common_nodes = set()
        for node in self.nodes:
            if node in tunnel.nodes:
                common_nodes.add(node)
        return common_nodes

    def check_collissions(self):
        if not self.check_self_collissions():
            print("Collides with itself")
            return False
        if not self.check_collisions_with_other_tunnels():
            print("Collides with other tunnel")
            return False
        return True

    def get_points_to_check_collisions_with_other_tunnels(
        self, min_dist, precision=None
    ):
        if precision is None:
            discretized_spline = self.spline.discretized[
                1
            ]  # select only the points of the discretized spline
        else:
            discretized_spline = self.spline.discretize(precision=precision)[1]
        if len(self.nodes[0].connected_nodes) > 1:
            indices_at_start = np.array(
                np.where(
                    np.linalg.norm(discretized_spline - self.nodes[0].xyz, axis=1)
                    < min_dist * 2
                )
            ).T
        else:
            indices_at_start = np.zeros([0, 1]).astype(np.int32)
        if len(self.nodes[-1].connected_nodes) > 1:
            indices_at_end = np.array(
                np.where(
                    np.linalg.norm(discretized_spline - self.nodes[-1].xyz, axis=1)
                    < min_dist * 2
                )
            ).T
        else:
            indices_at_end = np.zeros([0, 1]).astype(np.int32)
        elements_in_extremes = np.vstack([indices_at_start, indices_at_end])

        discretized_spline_without_tips = np.delete(
            discretized_spline, elements_in_extremes, axis=0
        )
        return discretized_spline_without_tips

    def check_collisions_with_other_tunnels(
        self, min_dist=MIN_DIST_OF_TUNNEL_COLLISSIONS
    ):
        discretized_spline_without_tips = (
            self.get_points_to_check_collisions_with_other_tunnels(min_dist=min_dist)
        )
        for tunnel in self._parent.tunnels:
            if tunnel is self:
                continue
            if any_point_close(
                discretized_spline_without_tips, tunnel.spline.discretized[1], min_dist
            ):
                return False
        return True

    def check_self_collissions(self, res=1, min_dist=6):
        spline = self.spline
        length = spline.length
        N = int(np.ceil(length / res))
        points = np.zeros((N, 3))
        ds = np.linspace(0, length, N)
        for n, d in enumerate(ds):
            p, v = spline(d)
            points[n, :] = p

        for n, p in enumerate(points):
            diffs = points - np.reshape(p, (1, 3))
            norms = np.linalg.norm(diffs, axis=1)
            indices = np.array(np.where(norms < min_dist))
            if np.max(np.abs(indices - n)) > min_dist / res + 1:
                return False
        return True, None

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


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = list()
        self._intersections = list()

    def remove_tunnel(self, tunnel):
        assert tunnel in self._tunnels, Exception(
            "Trying to remove a tunnel that is not in the tunnel network"
        )
        self._tunnels.remove(tunnel)

    def add_tunnel(self, tunnel: Tunnel):
        self._tunnels.append(tunnel)

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
            if len(node.tunnels) >= 2:
                self._intersections.append(node)
        return self._intersections
