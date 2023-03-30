import numpy as np
from perlin_noise import PerlinNoise
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.geometry import (
    get_two_perpendicular_vectors,
    get_close_points_indices,
    Point3D,
    warp_angle_2pi,
)
from subt_proc_gen.perlin import (
    CylindricalPerlinNoiseMapper,
    CylindricalPerlinNoiseMapperParms,
)
from enum import Enum
import time
from time import perf_counter_ns
from multiprocessing.pool import Pool
import logging as log

log.basicConfig(level=log.DEBUG)


def timeit(function, **args):
    start = perf_counter_ns()
    result = function(**args)
    end = perf_counter_ns()
    elapsed = (end - start) * 1e-9
    return result


class TunnelPtClGenParams:
    """Contains parameters for the generarion of a pointcloud around a single tunnel"""

    # Default params
    _default_dist_between_circles = 0.5
    _default_n_points_per_circle = 30
    _default_radius = 4
    _default_noise_relative_magnitude = 1
    _default_flatten_floor = True
    _default_fta_distance = 2
    # Random params
    _random_radius_interval = (1, 6)
    _random_noise_relative_magnitude_interval = (0.5, 0.1)
    _random_flatten_floor_probability = 0.5
    _random_fta_relative_distance_interval = [0, 1]

    @classmethod
    def from_defaults(cls):
        return TunnelPtClGenParams(
            dist_between_circles=cls._default_dist_between_circles,
            n_points_per_circle=cls._default_n_points_per_circle,
            radius=cls._default_radius,
            noise_relative_magnitude=cls._default_noise_relative_magnitude,
            flatten_floor=cls._default_flatten_floor,
            fta_distance=cls._default_fta_distance,
            perlin_params=CylindricalPerlinNoiseMapperParms.from_defaults(),
        )

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_interval[0],
            cls._random_radius_interval[1],
        )
        relative_magnitude = np.random.uniform(
            cls._random_noise_relative_magnitude_interval[0],
            cls._random_noise_relative_magnitude_interval[1],
        )
        flatten_floor = cls._random_flatten_floor_probability > np.random.random()
        fta_distance = radius * np.random.uniform(
            cls._random_noise_relative_magn_interval[0],
            cls._random_noise_relative_magn_interval[1],
        )
        return TunnelPtClGenParams(
            dist_between_circles=cls._default_dist_between_circles,
            n_points_per_circle=cls._default_n_points_per_circle,
            radius=radius,
            noise_relative_magnitude=relative_magnitude,
            flatten_floor=flatten_floor,
            fta_distance=fta_distance,
            perlin_params=CylindricalPerlinNoiseMapperParms.random(),
        )

    def __init__(
        self,
        dist_between_circles=None,
        n_points_per_circle=None,
        radius=None,
        noise_relative_magnitude=None,
        flatten_floor=None,
        fta_distance=None,
        perlin_params=None,
    ):
        assert not dist_between_circles is None
        assert not n_points_per_circle is None
        assert not radius is None
        assert not noise_relative_magnitude is None
        assert not flatten_floor is None
        assert not fta_distance is None
        assert not perlin_params is None
        self.dist_between_circles = dist_between_circles
        self.n_points_per_circle = n_points_per_circle
        self.radius = radius
        self.noise_relative_magnitude = noise_relative_magnitude
        self.flatter_floor = flatten_floor
        self.fta_distance = fta_distance
        self.perlin_params = perlin_params


class IntersectionPtClType(Enum):
    """Types of intersections to create
    - no_cavity: The points of the tunnels that are inside other tunnels are delted
    - cavity_along_one_tunnel: The pointcloud around one of the tunnels is inflated, and the
    """

    no_cavity = 1
    cavity_along_one_tunnel = 2
    spherical_cavity = 3


class IntersectionPtClGenParams:
    """Params that control how the pointcloud of an intersection is generated"""

    _default_radius = 10
    _default_type = IntersectionPtClType.no_cavity

    _random_radius_range = (5, 15)
    _random_type_choices = (
        IntersectionPtClType.no_cavity,
        IntersectionPtClType.spherical_cavity,
    )

    @classmethod
    def from_defaults(cls):
        return cls(radius=cls._default_radius, ptcl_type=cls._default_type)

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_range[0],
            cls._random_radius_range[1],
        )
        ptcl_type = np.random.choice(cls._random_type_choices)
        return cls(radius=radius, ptcl_type=ptcl_type)

    def __init__(self, radius=None, ptcl_type=None):
        assert not radius is None
        assert not ptcl_type is None
        self.radius = radius
        self.ptcl_type = ptcl_type


class TunnelNetworkPtClGenStrategies(Enum):
    """Different strategies to set the parameters of the ptcl
    generation of each of the tunnels"""

    all_random = 1
    all_default = 2
    pre_set_or_default = 3
    pre_set_or_random = 4


class TunnelNetworkPtClGenParams:
    """Params that control the the overall generation of the pointcloud of the
    complete tunnel network"""

    _default_ptcl_gen_strategy = TunnelNetworkPtClGenStrategies.all_default

    _random_ptcl_gen_strategy_choices = (
        TunnelNetworkPtClGenStrategies.all_default,
        TunnelNetworkPtClGenStrategies.all_random,
    )

    @classmethod
    def from_defaults(cls):
        return cls(ptcl_gen_strategy=cls._default_ptcl_gen_strategy)

    @classmethod
    def random(cls):
        return cls(
            ptcl_gen_strategy=np.random.choice(cls._random_ptcl_gen_strategy_choices)
        )

    def __init__(self, ptcl_gen_strategy: TunnelNetworkPtClGenStrategies):
        self.strategy = ptcl_gen_strategy


class TunnelNetworkMeshGenParams:
    """Params that control how the mesh is generated from the ptcl"""

    @classmethod
    def from_defaults(cls):
        return cls()

    @classmethod
    def random(cls):
        return cls()

    def __init__(self):
        pass


class TunnelNewtorkMeshGenerator:
    def __init__(
        self,
        tunnel_network: TunnelNetwork,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
        meshing_params: TunnelNetworkMeshGenParams,
    ):
        self._tunnel_network = tunnel_network
        self._ptcl_gen_params = ptcl_gen_params
        self.meshing_params = meshing_params
        self._ptcl_of_tunnels = dict()
        self._params_of_tunnels = dict()
        self._perlin_generator_of_tunnel = dict()
        self._ptcl_of_intersections = dict()
        self._axis_of_intersections = dict()
        self._params_of_intersections = dict()
        self._radius_of_intersections_for_tunnels = dict()

    def params_of_intersection(self, intersection) -> IntersectionPtClGenParams:
        return self._params_of_intersections[intersection]

    def params_of_tunnel(self, tunnel) -> TunnelPtClGenParams:
        return self._params_of_tunnels[tunnel]

    def _compute_intersection_ptcl(
        self,
        ptcl_gen_params: IntersectionPtClGenParams,
    ):
        raise NotImplementedError()

    def _set_params_of_each_tunnel_ptcl_gen(
        self, ptcl_gen_params: TunnelNetworkPtClGenParams
    ):
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.all_default:
            for tunnel in self._tunnel_network.tunnels:
                self._params_of_tunnels[tunnel] = TunnelPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.all_random:
            for tunnel in self._tunnel_network.tunnels:
                self._params_of_tunnels[tunnel] = TunnelPtClGenParams.random()
        elif (
            ptcl_gen_params.strategy
            == TunnelNetworkPtClGenStrategies.pre_set_or_default
        ):
            for tunnel in self._tunnel_network.tunnels:
                if tunnel in ptcl_gen_params.pre_set_tunnel_params:
                    self._params_of_tunnels[
                        tunnel
                    ] = ptcl_gen_params.pre_set_tunnel_params[tunnel]
                else:
                    self._params_of_tunnels[
                        tunnel
                    ] = TunnelPtClGenParams.from_defaults()
        elif (
            ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.pre_set_or_random
        ):
            for tunnel in self._tunnel_network.tunnels:
                if tunnel in ptcl_gen_params.pre_set_tunnel_params:
                    self._params_of_tunnels[
                        tunnel
                    ] = ptcl_gen_params.pre_set_tunnel_params[tunnel]
                else:
                    self._params_of_tunnels[tunnel] = TunnelPtClGenParams.random()

    def _set_params_of_each_intersection_ptcl_gen(
        self, ptcl_gen_params: TunnelNetworkPtClGenParams
    ):
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.all_default:
            for intersection in self._tunnel_network.intersections:
                self._params_of_intersections[
                    intersection
                ] = IntersectionPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.all_random:
            for intersection in self._tunnel_network.intersections:
                self._params_of_intersections[
                    intersection
                ] = IntersectionPtClGenParams.random()
        elif (
            ptcl_gen_params.strategy
            == TunnelNetworkPtClGenStrategies.pre_set_or_default
        ):
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.from_defaults()
        elif (
            ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.pre_set_or_random
        ):
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.random()

    def _set_perlin_mappers_of_tunnels(self):
        for tunnel in self._tunnel_network.tunnels:
            self._perlin_generator_of_tunnel[tunnel] = CylindricalPerlinNoiseMapper(
                sampling_scale=tunnel.spline.metric_length,
                params=self.params_of_tunnel(tunnel).perlin_params,
            )

    def _compute_tunnel_ptcls(self):
        for tunnel in self._tunnel_network.tunnels:
            self._ptcl_of_tunnels[tunnel] = ptcl_from_tunnel(
                tunnel=tunnel,
                perlin_mapper=self._perlin_generator_of_tunnel[tunnel],
                dist_between_circles=self.params_of_tunnel(tunnel).dist_between_circles,
                n_points_per_circle=self.params_of_tunnel(tunnel).n_points_per_circle,
                radius=self.params_of_tunnel(tunnel).radius,
                noise_magnitude=self.params_of_tunnel(tunnel).noise_relative_magnitude,
            )

    def _compute_point_at_tunnel_by_d_and_angle(
        self, tunnel: Tunnel, d, angle
    ) -> Point3D:
        angle = warp_angle_2pi(angle)
        ap, av = tunnel.spline(d)
        u, v = get_two_perpendicular_vectors(av)
        normal = u.xyz * np.sin(angle) + v.xyz * np.cos(angle)
        cylindrical_coords = np.reshape(
            np.array((d, angle * self.params_of_tunnel(tunnel).radius)), (1, 2)
        )
        noise = self.perlin_generator_of_tunnel(tunnel)(cylindrical_coords)
        radius = self.params_of_tunnel(tunnel).radius
        noise_magn = self.params_of_tunnel(tunnel).noise_relative_magnitude
        np_point = ap + normal * radius + normal * radius * noise_magn * noise
        return Point3D(np_point)

    def _compute_radius_of_intersections_for_all_tunnels(self):
        for intersection in self._tunnel_network.intersections:
            for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                max_rad = self.params_of_intersection(intersection).radius
                for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                    if tunnel_i is tunnel_j:
                        continue
                    current_rad = self.get_safe_radius_from_intersection_of_two_tunnels(
                        tunnel_i, tunnel_j, intersection
                    )
                    if current_rad > max_rad:
                        max_rad = current_rad
                self._radius_of_intersections_for_tunnels[
                    (intersection, tunnel_i)
                ] = max_rad

    def _separate_intersection_ptcl_from_tunnel_ptcls(self):
        for intersection in self._tunnel_network.intersections:
            tunnels_of_intersection = self._tunnel_network._tunnels_of_node[
                intersection
            ]
            ptcl_of_intersection = dict()
            axis_of_intersection = dict()
            for tunnel in tunnels_of_intersection:
                tunnel_ptcl = self._ptcl_of_tunnels[tunnel]
                radius = max(
                    self.params_of_intersection(intersection).radius,
                    self._radius_of_intersections_for_tunnels[(intersection, tunnel)],
                )
                indices_of_points_of_intersection = get_close_points_indices(
                    intersection.xyz, tunnel_ptcl, radius
                )
                tunnel_axis_points = tunnel.spline.discretize(
                    self.params_of_tunnel(tunnel).dist_between_circles
                )[1]
                indices_of_axis_points_of_intersection = get_close_points_indices(
                    intersection.xyz,
                    tunnel_axis_points,
                    radius,
                )
                ptcl_of_intersection[tunnel] = np.reshape(
                    tunnel_ptcl[indices_of_points_of_intersection, :], (-1, 3)
                )
                axis_of_intersection[tunnel] = np.reshape(
                    tunnel_axis_points[indices_of_axis_points_of_intersection, :],
                    (-1, 3),
                )
                self._ptcl_of_tunnels[tunnel] = np.delete(
                    tunnel_ptcl, indices_of_points_of_intersection, axis=0
                )
            self._ptcl_of_intersections[intersection] = ptcl_of_intersection
            self._axis_of_intersections[intersection] = axis_of_intersection

    def ptcl_of_intersections(self, intersection):
        return np.concatenate(
            [
                self._ptcl_of_intersections[intersection][tunnel]
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]
            ],
            axis=0,
        )

    def _compute_all_intersections_ptcl(self):
        for intersection in self._tunnel_network.intersections:
            params = self.params_of_intersection(intersection)
            ids_to_delete = dict()
            if params.ptcl_type == IntersectionPtClType.no_cavity:
                # Get the ids of the points to delete
                for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                    for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                        if tunnel_i is tunnel_j:
                            continue
                        ids_to_delete[tunnel_j] = points_inside_of_tunnel_section(
                            self._axis_of_intersections[intersection][tunnel_i],
                            self._ptcl_of_intersections[intersection][tunnel_i],
                            self._ptcl_of_intersections[intersection][tunnel_j],
                        )
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                    self._ptcl_of_intersections[intersection][tunnel] = np.reshape(
                        np.delete(
                            self._ptcl_of_intersections[intersection][tunnel],
                            ids_to_delete[tunnel],
                            axis=0,
                        ),
                        (-1, 3),
                    )

    def get_safe_radius_from_intersection_of_two_tunnels(
        self, tunnel_i: Tunnel, tunnel_j: Tunnel, intersection: Node, increment=2
    ):
        assert intersection in tunnel_i.nodes
        assert intersection in tunnel_j.nodes
        radius = self.params_of_intersection(intersection).radius
        if (intersection, tunnel_i) in self._radius_of_intersections_for_tunnels:
            radius = max(
                radius,
                self._radius_of_intersections_for_tunnels[(intersection, tunnel_i)],
            )
        if (intersection, tunnel_j) in self._radius_of_intersections_for_tunnels:
            radius = max(
                radius,
                self._radius_of_intersections_for_tunnels[(intersection, tunnel_j)],
            )
        ads, aps, avs = tunnel_i.spline.discretize(increment)
        dtaps = np.linalg.norm(aps - intersection.xyz, axis=1)
        intersect = True
        while intersect:
            ids = np.where(np.abs(dtaps - radius) < increment)
            for id in ids:
                d = ads[id].item(0)
                potia = self._compute_point_at_tunnel_by_d_and_angle(
                    tunnel=tunnel_i,
                    d=d,
                    angle=np.pi / 2,
                )
                potib = self._compute_point_at_tunnel_by_d_and_angle(
                    tunnel=tunnel_i, d=d, angle=np.pi / 2 * 3
                )
                if not self.is_point_inside_tunnel(
                    tunnel=tunnel_j, point=potia, threshold=0.3
                ) and not self.is_point_inside_tunnel(tunnel_j, potib, threshold=0.3):
                    intersect = False
                else:
                    intersect = True
                    radius += increment
        return radius + increment

    def compute_all(self):
        log.info("Setting parameters of tunnels")
        self._set_params_of_each_tunnel_ptcl_gen(self._ptcl_gen_params)
        log.info("Setting perlin mappers for tunnels")
        self._set_perlin_mappers_of_tunnels()
        log.info("Computing pointclouds of tunnels")
        self._compute_tunnel_ptcls()
        log.info("Setting paramters for intersections")
        self._set_params_of_each_intersection_ptcl_gen(self._ptcl_gen_params)
        log.info("Calcludating radius of intersections")
        self._compute_radius_of_intersections_for_all_tunnels()
        log.info("Separating the pointclouds of the tunnels for the intersections")
        self._separate_intersection_ptcl_from_tunnel_ptcls()
        log.info("Computing pointclouds of intersections")
        self._compute_all_intersections_ptcl()

    def perlin_generator_of_tunnel(
        self, tunnel: Tunnel
    ) -> CylindricalPerlinNoiseMapper:
        return self._perlin_generator_of_tunnel[tunnel]

    def multiprocesing_is_point_inside_tunnel(self, tp):
        return self.is_point_inside_tunnel(tp[0], tp[1])

    def is_point_inside_tunnel(self, tunnel: Tunnel, point, threshold=0.1):
        ad, ap, av = tunnel.spline.get_closest(point, precision=0.05)
        u, v = get_two_perpendicular_vectors(av)
        n = Point3D(point) - Point3D(ap)
        dist_to_axis = n.length
        if n.length == 0:
            return True
        n.normalize()
        proj_u = np.dot(n.xyz, u.xyz.T)
        proj_v = np.dot(n.xyz, v.xyz.T)
        angle = np.arctan2(proj_u, proj_v)
        if np.isnan(angle):
            angle = 0
        radius_of_tunnel_in_that_direction = np.linalg.norm(
            ap
            - self._compute_point_at_tunnel_by_d_and_angle(
                tunnel, ad.item(0), angle
            ).xyz
        )
        return radius_of_tunnel_in_that_direction > threshold + dist_to_axis


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(
    tunnel: Tunnel,
    perlin_mapper,
    dist_between_circles,
    n_points_per_circle,
    radius,
    noise_magnitude,
    d_min=None,
    d_max=None,
):
    ads, aps, avs = tunnel.spline.discretize(dist_between_circles, d_min, d_max)
    n_circles = aps.shape[0]
    angs = np.linspace(0, 2 * np.pi, n_points_per_circle)
    angss = np.concatenate([angs for _ in range(n_circles)])
    angss = np.reshape(angss, [-1, 1])
    apss = np.concatenate(
        [np.full((n_points_per_circle, 3), aps[i, :]) for i in range(n_circles)],
        axis=0,
    )
    us = np.zeros(avs.shape)
    vs = np.zeros(avs.shape)
    for i, av in enumerate(avs):
        u, v = get_two_perpendicular_vectors(av)
        us[i, :] = np.reshape(u.cartesian_unitary, -1)
        vs[i, :] = np.reshape(v.cartesian_unitary, -1)
    uss = np.concatenate(
        [np.full((n_points_per_circle, 3), us[i, :]) for i in range(n_circles)],
        axis=0,
    )
    vss = np.concatenate(
        [np.full((n_points_per_circle, 3), vs[i, :]) for i in range(n_circles)],
        axis=0,
    )
    normals = uss * np.sin(angss) + vss * np.cos(angss)
    dss = np.concatenate(
        [np.full((n_points_per_circle, 1), ads[i, :]) for i in range(n_circles)],
        axis=0,
    )
    archdss = angss * radius
    cylindrical_coords = np.concatenate([dss, archdss], axis=1)
    noise_to_add = perlin_mapper(cylindrical_coords)
    points = apss + normals * radius + normals * radius * noise_magnitude * noise_to_add
    return points


def points_inside_of_tunnel_section(axis_points, tunnel_points, points):
    # Tunnel axis size: Ax3
    # Tunnel points size: Tx3
    # POints size: Px3
    A = axis_points.shape[0]
    T = tunnel_points.shape[0]
    P = points.shape[0]
    # points axis dist: PxA
    # points tunnel dist: PxT
    dist_of_ps_to_axis = np.reshape(
        np.min(
            np.linalg.norm(
                np.ones((P, A, 3)) * np.reshape(axis_points, (1, A, 3))
                - np.reshape(points, (P, 1, 3)),
                axis=2,
            ),
            axis=1,
        ),
        (P, 1),
    )
    closest_t_to_ps = np.reshape(
        np.argmin(
            np.linalg.norm(
                np.ones((P, T, 3)) * np.reshape(tunnel_points, (1, T, 3))
                - np.reshape(points, (P, 1, 3)),
                axis=2,
            ),
            axis=1,
        ),
        (P, 1),
    )
    dists_of_ts_to_a = np.min(
        np.linalg.norm(
            np.ones((T, A, 3)) * np.reshape(axis_points, (1, A, 3))
            - np.reshape(tunnel_points, (T, 1, 3)),
            axis=2,
        ),
        axis=1,
    )
    return np.where(dist_of_ps_to_axis < dists_of_ts_to_a[closest_t_to_ps])
