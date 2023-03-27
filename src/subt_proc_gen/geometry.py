import numpy as np
import math as m
from scipy import interpolate
from subt_proc_gen.helper_functions import get_indices_close_to_point

###############################################################
# CLASSES
###############################################################


class Point3D:
    def __init__(self, coords):
        self.set_coords(coords)

    def set_coords(self, coords):
        if isinstance(coords, list) or isinstance(coords, tuple):
            assert len(coords) == 3
            self._coords = np.reshape(np.array(coords, dtype=np.double), (1, 3))
        elif isinstance(coords, np.ndarray):
            assert coords.size == 3
            self._coords = np.reshape(coords.astype(np.double), (1, 3))
        elif isinstance(coords, Point3D):
            self._coords = coords.xyz

    def __sub__(self, other):
        if isinstance(other, Vector3D):
            return Point3D(self.xyz - other.xyz)
        elif isinstance(other, Point3D):
            return Vector3D(self.xyz - other.xyz)

    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Point3D(self.xyz + other.xyz)
        elif isinstance(other, Point3D):
            raise TypeError("A Point cannot be added other Point")
        else:
            raise TypeError(f"Adding {type(other)} to Point not supported")

    def __eq__(self, other):
        if isinstance(other, Point3D):
            return np.all(self.xyz == other.xyz)
        return False

    @property
    def x(self):
        return self._coords[0, 0]

    @property
    def y(self):
        return self._coords[0, 1]

    @property
    def z(self):
        return self._coords[0, 2]

    @property
    def xyz(self):
        return self._coords


class Vector3D:
    @classmethod
    def from_inclination_yaw_length(cls, inclination, yaw, length):
        phi = inclination_to_phi(inclination)
        return Vector3D((length, yaw, phi), spherical_coords=True)

    def __init__(self, coords, spherical_coords=False):
        if isinstance(coords, Vector3D):
            self.__set_cartesian(coords.xyz)
        else:
            if spherical_coords:
                coords = format_coords(coords)
                self.__set_spherical(coords)
            else:
                coords = format_coords(coords)
                self.__set_cartesian(coords)

    def __abs__(self):
        return self._length

    def __add__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.xyz + other.xyz)
        elif isinstance(other, Vector3D):
            return Vector3D(self.xyz + other.xyz)
        else:
            raise NotImplementedError(f"Vector does not support adding {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Point3D):
            return Point3D(self.xyz - other.xyz)
        elif isinstance(other, Vector3D):
            return Vector3D(self.xyz - other.xyz)
        else:
            raise NotImplementedError(
                f"Vector does not support subtracting {type(other)}"
            )

    def __str__(self):
        return f"[{self.x},{self.x},{self.z}]"

    def __eq__(self, other):
        if isinstance(other, Vector3D):
            return np.all(np.isclose(self.xyz, other.xyz, rtol=1e-10))
        return False

    def __set_cartesian(self, coords):
        self._cartesian = coords
        self._length = np.linalg.norm(self._cartesian, axis=1)
        self._unit_cartesian_vector = self._cartesian / self.length
        self._spherical = np.reshape(
            np.array(
                (
                    self.length,
                    m.atan2(self.y, self.x),
                    m.atan2(self.xy_l, self.z),
                )
            ),
            (1, 3),
        )

    def __set_spherical(self, coords):
        self._spherical = coords
        self._length = coords[0, 0]
        self._cartesian = np.reshape(
            np.array(
                (
                    self.pho * m.sin(self.phi) * m.cos(self.theta),
                    self.pho * m.sin(self.phi) * m.sin(self.theta),
                    self.pho * m.cos(self.phi),
                )
            ),
            (1, 3),
        )

    def set_distance(self, new_length):
        self.__set_cartesian(self.xyz / self.length * new_length)

    @property
    def length(self):
        return self._length.item()

    @property
    def cartesian_unitary(self):
        return self._unit_cartesian_vector

    @property
    def x(self):
        return self._cartesian[0, 0].item()

    @property
    def y(self):
        return self._cartesian[0, 1].item()

    @property
    def z(self):
        return self._cartesian[0, 2].item()

    @property
    def xy(self):
        return self._cartesian[0, (0, 1)]

    @property
    def xyz(self):
        return self._cartesian

    @property
    def xy_l(self):
        return np.linalg.norm(self.xy, axis=0).item()

    @property
    def pho(self):
        return self._spherical[0, 0].item()

    @property
    def theta(self):
        return self._spherical[0, 1].item()

    @property
    def phi(self):
        return self._spherical[0, 2].item()

    @property
    def ptp(self):
        """pho_theta_phi"""
        return self._spherical

    @property
    def inclination(self):
        return phi_to_inclination(self.phi)


class Spline3D:
    """Wrapper around the scipy spline to
    interpolate a series of 3d points along x,y and z"""

    def __init__(self, points):
        for p in points:
            assert isinstance(p, Point3D)
        self._points = list(points)
        self._len = len(self._points)
        self._point_array = np.zeros(shape=[self._len, 3], dtype=np.double)
        self._dist_array = np.zeros(shape=[self._len, 1])
        for i, p in enumerate(self._points):
            self._point_array[i, :] = p.xyz
        for i in range(1, len(points)):
            self._dist_array[i, :] = self._dist_array[i - 1] + np.linalg.norm(
                self._point_array[i, :] - self._point_array[i - 1, :]
            )
        self._distance = self._dist_array[-1]
        self._degree = 3 if len(self._dist_array) > 3 else len(self._dist_array) - 1
        self.xspline = interpolate.splrep(
            self._dist_array, self._point_array[:, 0], k=self._degree
        )
        self.yspline = interpolate.splrep(
            self._dist_array, self._point_array[:, 1], k=self._degree
        )
        self.zspline = interpolate.splrep(
            self._dist_array, self._point_array[:, 2], k=self._degree
        )
        self._discretized = dict()

    def __call__(self, d):
        assert d >= 0 and d <= self._distance
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
        nd = int(np.ceil(self._distance / precision))
        # sampling distances
        ds = np.linspace(0, self._distance, nd)
        # sampled points and directions
        ps, vs = np.zeros([nd, 3]), np.zeros([nd, 3])
        for n, d in enumerate(ds):
            p, v = self(d)
            ps[n, :] = np.reshape(p, -1)
            vs[n, :] = np.reshape(v, -1)
        return ds, ps, vs

    def get_most_perpendicular_point_in_spline(
        self, point, threshold_distance, discretization_precision
    ):
        ds, ps, vs = self.discretize(discretization_precision)
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

    def distance_matrix(self, other, discretization_precision):
        assert isinstance(other, Spline3D)
        self_d = self.discretize(discretization_precision)[1]
        other_d = other.discretize(discretization_precision)[1]
        distance_matrix = np.ones([])

    def collides(self, other, threshold_distance):
        assert isinstance(other, Spline3D)


###############################################################
# FUNCTIONS
###############################################################
def phi_to_inclination(phi):
    return m.pi / 2 - phi


def inclination_to_phi(inclination):
    return -inclination + m.pi / 2


def format_coords(coords):
    if isinstance(coords, np.ndarray):
        if coords.shape == (1, 3):
            return coords
        else:
            assert coords.size == 3
            return np.reshape(coords, (1, 3)).astype(np.double)
    if (
        isinstance(coords, list)
        or isinstance(coords, tuple)
        or isinstance(coords, set)
        or isinstance(coords, frozenset)
    ):
        assert len(coords) == 3
        return np.reshape(np.array(coords, dtype=np.double), (1, 3))


###############################################################
# TESTING
###############################################################
def test1():
    p1 = Point3D(np.random.random([1, 3]))
    p2 = Point3D(np.random.random([1, 3]))
    v2 = p1 - p2
    assert isinstance(v2, Vector3D)
    p3 = p2 + v2
    assert isinstance(p3, Point3D)
    assert p3 == p1


def test2():
    p1 = Point3D(np.random.random([1, 3]))
    p2 = Point3D(np.random.random([1, 3]))
    v1 = p1 - p2
    v2 = Vector3D(v1.ptp, spherical_coords=True)
    assert v2 == v1


def main():
    test1()
    test2()


if __name__ == "__main__":
    for _ in range(200):
        main()
