import numpy as np
from perlin_noise import PerlinNoise
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.geometry import (
    get_two_perpendicular_vectors,
    get_close_points_indices,
    Point3D,
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


class OctaveToMagnitudeScalingTypes(Enum):
    inverse = 1  # M = 1/O * c1
    linear = 2  # M = O*c1
    inverse_root = 3  # M = 1/sqrt(O) * c1
    constant = 4  # M = Ms[idx_of(O)]
    exponential = 5  # M = (c2)^(idx_of_(O)) * c1


class OctaveProgressionType(Enum):
    exponential = 1


class CylindricalPerlinNoiseMapperParms:
    """Contains the parameters that controll the perlin cylindrical mapper. This also includes
    parameters that controll how the octave generator and the octave to magnitude mapper works"""

    _default_roughness = 0.01
    _default_n_layers = 5
    _default_octave_progression = OctaveProgressionType.exponential
    _default_octave_progression_consts = (2,)
    _default_octave_to_magnitude_scaling = OctaveToMagnitudeScalingTypes.inverse_root
    _default_octave_to_magnitude_consts = (1.0,)

    _random_roughness_range = (0.3, 0.0001)
    _random_n_layers_range = (1, 6)
    _random_octave_progression_types = (OctaveProgressionType.exponential,)
    _random_octave_progression_const_ranges = ((0.5, 0.5),)
    _random_octave_to_magnitude_scaling_types = (OctaveToMagnitudeScalingTypes.inverse,)
    _random_octave_to_magnitude_const_ranges = ((1, 1),)

    @classmethod
    def from_defaults(cls):
        return cls(
            roughness=cls._default_roughness,
            n_layers=cls._default_n_layers,
            octave_progression=cls._default_octave_progression,
            octave_progression_consts=cls._default_octave_progression_consts,
            octave_to_magnitude_scaling=cls._default_octave_to_magnitude_scaling,
            octave_to_magnitude_consts=cls._default_octave_to_magnitude_consts,
        )

    @classmethod
    def random(cls):
        return cls(
            roughness=np.random.uniform(
                cls._random_roughness_range[0],
                cls._random_roughness_range[1],
            ),
            n_layers=np.random.random_integers(
                cls._random_n_layers_range[0],
                cls._random_n_layers_range[1],
            ),
            octave_progression=np.random.choice(cls._random_octave_progression_types),
            octave_progression_consts=tuple(
                [
                    np.random.uniform(p[0], p[1])
                    for p in cls._random_octave_progression_const_ranges
                ]
            ),
            octave_to_magnitude_scaling=np.random.choice(
                cls._random_octave_to_magnitude_scaling_types
            ),
            octave_to_magnitude_consts=tuple(
                [
                    np.random.uniform(p[0], p[1])
                    for p in cls._random_octave_to_magnitude_const_ranges
                ]
            ),
        )

    def __init__(
        self,
        roughness,
        n_layers,
        octave_progression,
        octave_progression_consts,
        octave_to_magnitude_scaling,
        octave_to_magnitude_consts,
    ):
        self.roughness = roughness
        self.n_layers = n_layers
        self.octave_progression = octave_progression
        self.octave_progression_consts = octave_progression_consts
        self.octave_to_magnitude_scaling = octave_to_magnitude_scaling
        self.octave_to_magnitude_consts = octave_to_magnitude_consts


class OctaveProgressionGenerator:
    """Creates a progresion of numbers that is used to stablish the octaves of the different
    layers of the perlin generator"""

    def __init__(self, type: OctaveProgressionType, constants):
        self._generation_type = type
        self._constants = constants

    def __call__(self, n_octaves):
        # This should alwas return a decreasing set of floats starting at 1.0
        if self._generation_type == OctaveProgressionType.exponential:
            return self._exponential(n_octaves)

    def _exponential(self, n_octaves):
        assert len(self._constants) == 1
        base = self._constants[0]
        octaves = []
        for exp in range(0, n_octaves):
            octaves.append(base**exp)
        return octaves


class PerlinMagnitudesGenerator:
    """Controls the relationship between the octave of a layer and it's magnitude"""

    def __init__(self, type: OctaveProgressionType, constants):
        self._generation_type = type
        self._constants = constants

    def __call__(self, octaves):
        if self._generation_type == OctaveToMagnitudeScalingTypes.inverse:
            return [
                self._inverse(n_octave, octave)
                for n_octave, octave in enumerate(octaves)
            ]
        elif self._generation_type == OctaveToMagnitudeScalingTypes.inverse_root:
            return [
                self._inverse_root(n_octave, octave)
                for n_octave, octave in enumerate(octaves)
            ]

    def _inverse(self, n_octave, octave):
        assert len(self._constants) == 1
        mult = self._constants[0]
        return 1 / octave * mult

    def _inverse_root(self, n_octave, octave):
        assert len(self._constants) == 1
        mult = self._constants[0]
        return (1 / octave**0.5) * mult


class CylindricalPerlinNoiseMapper:
    def __init__(
        self, sampling_scale, params: CylindricalPerlinNoiseMapperParms, seed=None
    ):
        if seed is None:
            self._seed = time.time_ns()
        self._params = params
        self._sampling_scale = sampling_scale
        self._octave_progression = OctaveProgressionGenerator(
            type=self._params.octave_progression,
            constants=self._params.octave_progression_consts,
        )(self._params.n_layers)
        self._octaves = [
            base_octave * self._sampling_scale * self._params.roughness
            for base_octave in self._octave_progression
        ]
        self._magnitudes = PerlinMagnitudesGenerator(
            self._params.octave_to_magnitude_scaling,
            self._params.octave_to_magnitude_consts,
        )(self._octave_progression)
        self._noise_generators = [
            PerlinNoise(octaves=octave, seed=seed) for octave in self._octaves
        ]

    def _noise_of_scaled_coords(self, scaled_coords):
        return sum(
            [
                noise(scaled_coords) * mag
                for noise, mag in zip(self._noise_generators, self._magnitudes)
            ]
        )

    def __call__(self, coords):
        if type(coords) in [tuple, list]:
            coords = np.array(coords)
        scaled_coords = coords / self._sampling_scale
        pool = Pool()
        noises = pool.map(self._noise_of_scaled_coords, scaled_coords)
        pool.close()
        return np.array(noises)


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

    def _compute_all_tunnels_ptcl(self):
        for tunnel in self._tunnel_network.tunnels:
            (
                self._ptcl_of_tunnels[tunnel],
                self._perlin_generator_of_tunnel[tunnel],
            ) = ptcl_from_tunnel(tunnel, self._params_of_tunnels[tunnel])

    def _compute_point_at_tunnel_by_cylindrical_coords(
        self, tunnel: Tunnel, d, angle
    ) -> Point3D:
        ap, av = tunnel.spline(d)
        u, v = get_two_perpendicular_vectors(av)
        normal = u.cartesian_unitary * np.sin(angle) + v.cartesian_unitary * np.cos(
            angle
        )
        noise = self.perlin_generator_of_tunnel(tunnel)(
            np.reshape(np.array((d, angle)), (1, 2))
        )
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
            ptcl_of_intersection = np.zeros((0, 3))
            for tunnel in tunnels_of_intersection:
                tunnel_ptcl = self._ptcl_of_tunnels[tunnel]
                radius = max(
                    self.params_of_intersection(intersection).radius,
                    self._radius_of_intersections_for_tunnels[(intersection, tunnel)],
                )
                indices_of_points_of_intersection = get_close_points_indices(
                    intersection.xyz, tunnel_ptcl, radius
                )
                a = np.reshape(
                    tunnel_ptcl[indices_of_points_of_intersection, :], (-1, 3)
                )
                ptcl_of_intersection = np.concatenate(
                    [ptcl_of_intersection, a],
                    axis=0,
                )
                self._ptcl_of_tunnels[tunnel] = np.delete(
                    tunnel_ptcl, indices_of_points_of_intersection, axis=0
                )
            self._ptcl_of_intersections[intersection] = ptcl_of_intersection

    def _compute_all_intersections_ptcl(self):
        raise NotImplementedError()

    def get_safe_radius_from_intersection_of_two_tunnels(
        self, tunnel_i: Tunnel, tunnel_j: Tunnel, intersection: Node, increment=0.5
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
                potia = self._compute_point_at_tunnel_by_cylindrical_coords(
                    tunnel=tunnel_i,
                    d=d,
                    angle=np.pi / 2,
                )
                potib = self._compute_point_at_tunnel_by_cylindrical_coords(
                    tunnel=tunnel_i, d=d, angle=np.pi / 2 * 3
                )
                if not self.is_point_inside_tunnel(
                    tunnel=tunnel_j, point=potia
                ) and not self.is_point_inside_tunnel(tunnel_j, potib):
                    intersect = False
                else:
                    intersect = True
                    radius += increment
        return radius + increment

    def compute_all(self):
        log.info("Setting parameters of tunnels")
        self._set_params_of_each_tunnel_ptcl_gen(self._ptcl_gen_params)
        log.info("Computing pointclouds of tunnels")
        self._compute_all_tunnels_ptcl()
        log.info("Setting paramters for intersections")
        self._set_params_of_each_intersection_ptcl_gen(self._ptcl_gen_params)
        log.info("Calcludating radius of intersections")
        self._compute_radius_of_intersections_for_all_tunnels()
        log.info("Separating the pointclouds of the tunnels for the intersections")
        self._separate_intersection_ptcl_from_tunnel_ptcls()
        # self._compute_all_intersections_ptcl()

    def perlin_generator_of_tunnel(
        self, tunnel: Tunnel
    ) -> CylindricalPerlinNoiseMapper:
        return self._perlin_generator_of_tunnel[tunnel]

    def is_point_inside_tunnel(self, tunnel: Tunnel, point, threshold=0.1):
        ad, ap, av = tunnel.spline.get_closest(point)
        u, v = get_two_perpendicular_vectors(av)
        n = Point3D(point) - Point3D(ap)
        dist_to_axis = n.length
        n.normalize()
        angle = np.arctan2(u.x / n.x - u.y / n.y, v.y / n.y - v.x / n.x)
        radius_of_tunnel_in_that_direction = np.linalg.norm(
            ap
            - self._compute_point_at_tunnel_by_cylindrical_coords(
                tunnel, ad.item(0), angle
            ).xyz
        )
        return radius_of_tunnel_in_that_direction > threshold + dist_to_axis


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(tunnel: Tunnel, params: TunnelPtClGenParams):
    ds, ps, avs = tunnel.spline.discretize(params.dist_between_circles)
    n_circles = ps.shape[0]
    angs = np.linspace(0, 2 * np.pi, params.n_points_per_circle)
    angss = np.concatenate([angs for _ in range(n_circles)])
    angss = np.reshape(angss, [-1, 1])
    ptcl = np.zeros((ps.shape[0] * len(angs), 3))
    pss = np.concatenate(
        [np.full((params.n_points_per_circle, 3), ps[i, :]) for i in range(n_circles)],
        axis=0,
    )
    us = np.zeros(avs.shape)
    vs = np.zeros(avs.shape)
    for i, av in enumerate(avs):
        u, v = get_two_perpendicular_vectors(av)
        us[i, :] = np.reshape(u.cartesian_unitary, -1)
        vs[i, :] = np.reshape(v.cartesian_unitary, -1)
    uss = np.concatenate(
        [np.full((params.n_points_per_circle, 3), us[i, :]) for i in range(n_circles)],
        axis=0,
    )
    vss = np.concatenate(
        [np.full((params.n_points_per_circle, 3), vs[i, :]) for i in range(n_circles)],
        axis=0,
    )
    normals = uss * np.sin(angss) + vss * np.cos(angss)
    points = pss + normals * params.radius
    dss = np.concatenate(
        [np.full((params.n_points_per_circle, 3), ds[i, :]) for i in range(n_circles)],
        axis=0,
    )
    archdss = angss * params.radius
    cylindrical_coords = np.concatenate([dss, archdss], axis=1)
    perlin_generator = CylindricalPerlinNoiseMapper(
        sampling_scale=tunnel.spline.metric_length,
        params=params.perlin_params,
    )
    noise_to_add = perlin_generator(cylindrical_coords)
    points += normals * params.radius * params.noise_relative_magnitude * noise_to_add
    return points, perlin_generator
