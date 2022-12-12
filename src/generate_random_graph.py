import shapely
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import interpolate
import math

from helper_functions import *

from PARAMS import MAX_SEGMENT_INCLINATION, MIN_ANGLE_FOR_INTERSECTIONS, SPLINE_PLOT_PRECISSION
from tunnel import TunnelParams, Tunnel
from graph import Graph, Node

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
