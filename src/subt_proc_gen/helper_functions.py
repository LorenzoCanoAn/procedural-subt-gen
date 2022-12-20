import math
import numpy as np

def angles_to_vector(angles):
    th, ph = angles
    xy = math.cos(ph)
    x = xy*math.cos(th)
    y = xy*math.sin(th)
    z = math.sin(ph)
    return np.array((x, y, z))


def vector_to_angles(vector):
    x, y, z = vector
    ph = math.atan2(z, x**2+y**2)
    th = math.atan2(y, x)
    return th, ph


def warp_angle_2pi(angle):
    while angle < 0:
        angle += 2*math.pi
    return angle % (2*math.pi)


def warp_angle_pi(angle):
    new_angle = warp_angle_2pi(angle)
    if new_angle > np.pi:
        new_angle -= 2*math.pi
    return new_angle

def horizontally_close(array_of_points, points, threshold):
    """Retu"""