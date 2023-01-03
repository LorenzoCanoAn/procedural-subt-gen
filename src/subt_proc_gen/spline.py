from scipy import interpolate
import numpy as np
from subt_proc_gen.PARAMS import MIN_DIST_OF_MESH_POINTS
from subt_proc_gen.helper_functions import get_indices_close_to_point


class Spline3D:
    """Wrapper around the scipy spline to
    interpolate a series of 3d points along x,y and z"""

    def __init__(self, points, precision=MIN_DIST_OF_MESH_POINTS):
        self.points = np.array(points)
        self.precision = precision
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
        self._discretized = None
        self._precision_discretized = None

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

    def discretize(self, precision):
        # number of sampling points
        nd = int(np.ceil(self.length / precision))
        # sampling distances
        ds = np.linspace(0, self.length, nd)
        # sampled points and directions
        ps, vs = np.zeros([nd, 3]), np.zeros([nd, 3])
        for n, d in enumerate(ds):
            ps[n], vs[n] = self(d)
        return ds, ps, vs

    @property
    def discretized(self):
        if self._discretized is None:
            self._discretized = self.discretize(self.precision)
        return self._discretized

    @property
    def precision_discretized(self):
        if self._precision_discretized is None:
            self._precision_discretized = self.discretize(0.05)
        return self._precision_discretized

    def get_most_perpendicular_point_in_spline(self, point, threshold_distance):
        ds, ps, vs = self.precision_discretized
        # get only the points of the spline close to point
        ids = get_indices_close_to_point(
            ps, point, threshold_distance, horizontal_distance=False
        )
        if len(ids) == 0:
            return None
        ps, vs = ps[ids, :], vs[ids, :]
        # get the vectors that go from the point to the spline points
        vpps = point - ps
        dists = np.linalg.norm(vpps, axis=1)
        vpps /= np.reshape(dists, [-1, 1])
        # get the cross product between the spline orientation at each
        # of the spline point and the vector that goes from the spline
        # point to the point
        cps = np.cross(vpps, vs, axis=1)
        cpns = np.linalg.norm(cps, axis=1)
        i = np.argmax(cpns)
        if max(cpns) < 0.95:
            return None
        return (
            ds[i],
            ps[i, :],
            vs[i, :],
        )
