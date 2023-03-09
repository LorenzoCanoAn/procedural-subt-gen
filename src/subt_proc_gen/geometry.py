import numpy as np
import math as m

###############################################################
# CLASSES
###############################################################


class Point:
    def __init__(self, coords):
        self.set_coords(coords)

    def set_coords(self, coords):
        if isinstance(coords, list) or isinstance(coords, tuple):
            assert len(coords) == 3
            self._coords = np.reshape(np.array(coords, dtype=np.double), (1, 3))
        elif isinstance(coords, np.ndarray):
            assert coords.size == 3
            self._coords = np.reshape(coords.astype(np.double), (1, 3))

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Point(self.xyz - other.xyz)
        elif isinstance(other, Point):
            return Vector(self.xyz - other.xyz)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Point(self.xyz + other.xyz)
        elif isinstance(other, Point):
            raise TypeError("A Point cannot be added other Point")
        else:
            raise TypeError(f"Adding {type(other)} to Point not supported")

    def __eq__(self, other):
        if isinstance(other, Point):
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


class Vector:
    def __init__(self, coords, spherical_coords=False):
        if spherical_coords:
            self.__set_spherical(coords)
        else:
            self.__set_cartesian(coords)

    def __abs__(self):
        return self._length

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.xyz + other.xyz)
        elif isinstance(other, Vector):
            return Vector(self.xyz + other.xyz)
        else:
            raise NotImplementedError(f"Vector does not support adding {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.xyz - other.xyz)
        elif isinstance(other, Vector):
            return Vector(self.xyz - other.xyz)
        else:
            raise NotImplementedError(
                f"Vector does not support subtracting {type(other)}"
            )

    def __eq__(self, other):
        if isinstance(other, Vector):
            return np.all(np.isclose(self.xyz, other.xyz, rtol=1e-10))
        return False

    def __set_cartesian(self, coords):
        self._cartesian = coords
        self._length = np.linalg.norm(self._cartesian, axis=1)
        self._unit_vector = self._cartesian / self.length
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

    @property
    def length(self):
        return self._length.item()

    @property
    def unitary(self):
        return self._unit_vector

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


###############################################################
# FUNCTIONS
###############################################################
def phi_to_inclination(phi):
    return m.pi / 2 - phi


def inclination_to_phi(inclination):
    return -inclination + m.pi / 2


###############################################################
# TESTING
###############################################################
def test1():
    p1 = Point(np.random.random([1, 3]))
    p2 = Point(np.random.random([1, 3]))
    v2 = p1 - p2
    assert isinstance(v2, Vector)
    p3 = p2 + v2
    assert isinstance(p3, Point)
    assert p3 == p1


def test2():
    p1 = Point(np.random.random([1, 3]))
    p2 = Point(np.random.random([1, 3]))
    v1 = p1 - p2
    v2 = Vector(v1.ptp, spherical_coords=True)
    assert v2 == v1


def main():
    test1()
    test2()


if __name__ == "__main__":
    for _ in range(200):
        main()
