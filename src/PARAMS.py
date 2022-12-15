from math import pi
from numpy import deg2rad, array
# GRAPH GENERATION PARAMETERS
MAX_SEGMENT_INCLINATION = 10/180 * pi  # rad
MIN_ANGLE_FOR_INTERSECTIONS = deg2rad(50)
SPLINE_PLOT_PRECISSION = 0.3

# MESH GENERATION PARAMETERS
MIN_DIST_OF_MESH_POINTS = 0.1  # meters
N_ANGLES_PER_CIRCLE = 10
TUNNEL_AVG_RADIUS = 3 # meters

COLORS = (
    array((128,128,0))/255.,
    array((0,20,255))/255.,
    array((75,0,130))/255.,
    array((255,255,0))/255.,
    array((75,0,130))/255.,
    array((255,69,0))/255.,
)