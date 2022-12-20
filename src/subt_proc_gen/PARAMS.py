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
    array((255,0,0))/255.,
    array((255,255,0))/255.,
    array((255,0,255))/255.,
    array((0,255,0))/255.,
    array((0,255,255))/255.,
    array((126,0,0))/255.,
    array((126,126,0))/255.,
    array((126,0,126))/255.,
    array((0,126,0))/255.,
    array((0,126,126))/255.,
    array((0,0,126))/255.,
    array((75,0,130))/255.,
    array((255,69,0))/255.,
    array((128,128,0))/255.,
)