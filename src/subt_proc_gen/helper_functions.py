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


def get_indices_of_points_below_cylinder(points, center, radius):
    points_xy = points[:, :2]
    below_z = points[:, -1] < center[-1]
    differences = points_xy - center[:2]
    distances = np.linalg.norm(differences, axis=1)
    horizontaly_close = distances < radius
    return np.array(np.where(horizontaly_close & below_z)).flatten()


def angle_between_angles(a1, a2):
    if a1 is None or a2 is None:
        return np.pi
    return abs(warp_angle_pi(a2 - a1))


def get_indices_close_to_point(
    points: np.ndarray, point: np.ndarray, threshold_distance, horizontal_distance=True
):
    """points should have a 3x1 dimmension.
    - points: Nx3 array
    - point: 1x3 array
    - threshold distance: if a point of points is closer to point than this distance, it will be selected
    - horizontal distance: if this parameter is set to true, the distance between points and the point will only be measured in the xy plane"""
    if horizontal_distance:
        points_xy = points[:, :2]
        differences = points_xy - np.reshape(point.flatten()[:2], [1, 2])
    else:
        differences = points - np.reshape(point.flatten(), [1, 3])
    distances = np.linalg.norm(differences, axis=1)
    return np.array(np.where(distances < threshold_distance)).flatten()
