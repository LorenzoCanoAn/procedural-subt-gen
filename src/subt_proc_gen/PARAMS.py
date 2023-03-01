from math import pi
from numpy import deg2rad, array

# GRAPH GENERATION PARAMETERS
MAX_SEGMENT_INCLINATION = deg2rad(90)  # rad
MIN_ANGLE_FOR_INTERSECTIONS = deg2rad(50)
SPLINE_PLOT_PRECISSION = 0.3
MIN_DIST_OF_TUNNEL_COLLISSIONS = 5
# MESH GENERATION PARAMETERS
MIN_DIST_OF_MESH_POINTS = 1  # meters
N_ANGLES_PER_CIRCLE = 50
TUNNEL_AVG_RADIUS = 2  # meters

# Distance at which points are considered part of an intersection
INTERSECTION_DISTANCE = 10

COLORS = (
    array((255, 0, 0)) / 255.0,
    array((255, 255, 0)) / 255.0,
    array((255, 0, 255)) / 255.0,
    array((0, 255, 0)) / 255.0,
    array((0, 255, 255)) / 255.0,
    array((126, 0, 0)) / 255.0,
    array((126, 126, 0)) / 255.0,
    array((126, 0, 126)) / 255.0,
    array((0, 126, 0)) / 255.0,
    array((0, 126, 126)) / 255.0,
    array((0, 0, 126)) / 255.0,
    array((75, 0, 130)) / 255.0,
    array((255, 69, 0)) / 255.0,
    array((128, 128, 0)) / 255.0,
)
