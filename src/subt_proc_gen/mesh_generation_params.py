import numpy as np
from enum import Enum
from subt_proc_gen.perlin import (
    CylindricalPerlinNoiseMapperParms,
    SphericalPerlinNoiseMapperParms,
)


class TunnelPtClGenParams:
    """Contains parameters for the generarion of a pointcloud around a single tunnel"""

    # Default params
    _default_dist_between_circles = 0.5
    _default_n_points_per_circle = 30
    _default_radius = 4
    _default_noise_relative_magnitude = 1
    _default_noise_along_angle_multiplier = 1.5
    _default_flatten_floor = True
    _default_fta_distance = 2
    # Random params
    _random_radius_interval = (1, 6)
    _random_noise_relative_magnitude_interval = (0.5, 0.1)
    _random_noise_along_angle_multiplier_interval = (1, 2)
    _random_flatten_floor_probability = 0.5
    _random_fta_relative_distance_interval = [0, 1]

    @classmethod
    def from_defaults(cls):
        return TunnelPtClGenParams(
            dist_between_circles=cls._default_dist_between_circles,
            n_points_per_circle=cls._default_n_points_per_circle,
            radius=cls._default_radius,
            noise_relative_magnitude=cls._default_noise_relative_magnitude,
            noise_along_angle_multiplier=cls._default_noise_along_angle_multiplier,
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
        noise_along_angle_multiplier = np.random.uniform(
            cls._random_noise_along_angle_multiplier_interval[0],
            cls._random_noise_along_angle_multiplier_interval[1],
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
            noise_along_angle_multiplier=noise_along_angle_multiplier,
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
        noise_along_angle_multiplier=None,
        flatten_floor=None,
        fta_distance=None,
        perlin_params=None,
        perlin_weight_angle=np.deg2rad(40),
    ):
        assert not dist_between_circles is None
        assert not n_points_per_circle is None
        assert not radius is None
        assert not noise_relative_magnitude is None
        assert not noise_along_angle_multiplier is None
        assert not flatten_floor is None
        assert not fta_distance is None
        assert not perlin_params is None
        self.dist_between_circles = dist_between_circles
        self.n_points_per_circle = n_points_per_circle
        self.radius = radius
        self.noise_relative_magnitude = noise_relative_magnitude
        self.noise_along_angle_multiplier = noise_along_angle_multiplier
        self.flatter_floor = flatten_floor
        self.fta_distance = fta_distance
        self.perlin_params = perlin_params
        self.perlin_weight_by_angle = perlin_weight_angle


class IntersectionPtClType(Enum):
    """Types of intersections to create
    - no_cavity: The points of the tunnels that are inside other tunnels are delted
    - cavity_along_one_tunnel: The pointcloud around one of the tunnels is inflated, and the
    """

    no_cavity = 1
    spherical_cavity = 2
    cavity_along_one_tunnel = 3


class IntersectionPtClGenParams:
    """Params that control how the pointcloud of an intersection is generated"""

    _default_radius = 10
    _default_type = IntersectionPtClType.no_cavity
    _default_point_density = 3  # Points per sqare meter

    _random_radius_range = (5, 15)
    _random_type_choices = (
        IntersectionPtClType.no_cavity,
        IntersectionPtClType.spherical_cavity,
    )

    @classmethod
    def from_defaults(cls):
        return cls(
            radius=cls._default_radius,
            ptcl_type=cls._default_type,
            perlin_params=SphericalPerlinNoiseMapperParms.from_defaults(),
        )

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_range[0],
            cls._random_radius_range[1],
        )
        ptcl_type = np.random.choice(cls._random_type_choices)
        return cls(
            radius=radius,
            ptcl_type=ptcl_type,
            perlin_params=SphericalPerlinNoiseMapperParms.random(),
        )

    def __init__(self, radius=None, ptcl_type=None, perlin_params=None):
        assert not radius is None
        assert not ptcl_type is None
        assert not perlin_params is None
        self.radius = radius
        self.ptcl_type = ptcl_type
        self._perlin_params = perlin_params

    @property
    def perlin_params(self) -> SphericalPerlinNoiseMapperParms:
        return self._perlin_params


class TunnelNetworkPtClGenStrategies(Enum):
    """Different strategies to set the parameters of the ptcl
    generation of each of the tunnels"""

    random = 1
    default = 2


class TunnelNetworkPtClGenParams:
    """Params that control the the overall generation of the pointcloud of the
    complete tunnel network"""

    _default_ptcl_gen_strategy = TunnelNetworkPtClGenStrategies.default
    _default_perlin_weight_by_angle = np.deg2rad(40)

    _random_ptcl_gen_strategy_choices = (
        TunnelNetworkPtClGenStrategies.default,
        TunnelNetworkPtClGenStrategies.random,
    )

    @classmethod
    def from_defaults(cls, pre_set_tunnel_params=dict()):
        return cls(
            ptcl_gen_strategy=cls._default_ptcl_gen_strategy,
            perlin_weight_by_angle=cls._default_perlin_weight_by_angle,
            pre_set_tunnel_params=pre_set_tunnel_params,
        )

    @classmethod
    def random(cls, pre_set_tunnel_params=dict()):
        return cls(
            ptcl_gen_strategy=np.random.choice(
                ptcl_gen_strategy=np.random.choice(
                    cls._random_ptcl_gen_strategy_choices
                ),
                perlin_weight_by_angle=cls._default_perlin_weight_by_angle,
                pre_set_tunnel_params=pre_set_tunnel_params,
            )
        )

    def __init__(
        self,
        ptcl_gen_strategy: TunnelNetworkPtClGenStrategies,
        perlin_weight_by_angle,
        pre_set_tunnel_params,
    ):
        self.strategy = ptcl_gen_strategy
        self.perlin_weight_by_angle = perlin_weight_by_angle
        self.pre_set_tunnel_params = pre_set_tunnel_params


class MeshingApproaches(Enum):
    poisson = 1


class TunnelNetworkMeshGenParams:
    """Params that control how the mesh is generated from the ptcl"""

    _default_meshing_approach = MeshingApproaches.poisson
    _default_poisson_depth = 11

    @classmethod
    def from_defaults(cls):
        return cls(
            meshing_approach=cls._default_meshing_approach,
            poisson_depth=cls._default_poisson_depth,
        )

    def __init__(self, meshing_approach=None, poisson_depth=None):
        assert not meshing_approach is None
        if meshing_approach == MeshingApproaches.poisson:
            assert not poisson_depth is None
        self.meshing_approach = meshing_approach
        self.poisson_depth = poisson_depth
