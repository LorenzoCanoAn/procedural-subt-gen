import math
import numpy as np


def angles_to_vector(angles):
    th, ph = angles
    xy = math.cos(ph)
    x = xy * math.cos(th)
    y = xy * math.sin(th)
    z = math.sin(ph)
    return np.array((x, y, z))


def vector_to_angles(vector):
    x, y, z = vector
    ph = math.atan2(z, x**2 + y**2)
    th = math.atan2(y, x)
    return th, ph


def warp_angle_2pi(angle):
    while angle < 0:
        angle += 2 * math.pi
    return angle % (2 * math.pi)


def warp_angle_pi(angle):
    new_angle = warp_angle_2pi(angle)
    if new_angle > np.pi:
        new_angle -= 2 * math.pi
    return new_angle


def gen_cylinder_around_point(
    center_coordinates: np.ndarray, height, radius, n_points=1000
):
    center_coordinates = np.reshape(center_coordinates.flatten(), [1, -1])
    angle = np.random.uniform(0, 2 * np.pi, n_points)
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = np.random.uniform(-height / 2, height / 2, n_points)
    vector = np.array((x, y, z)).T
    return center_coordinates + vector
