import numpy as np
from perlin_noise import PerlinNoise
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from enum import Enum
import time
from multiprocessing.pool import Pool


class OctaveToMagnitudeScalingTypes(Enum):
    inverse = 1  # M = 1/O * c1
    linear = 2  # M = O*c1
    inverse_root = 3  # M = 1/sqrt(O) * c1
    constant = 4  # M = Ms[idx_of(O)]
    exponential = 5  # M = (c2)^(idx_of_(O)) * c1


class OctaveProgressionType(Enum):
    exponential = 1


class CylindricalPerlinNoiseMapperParms:
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


class MagnitudesGenerator:
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
        self._magnitudes = MagnitudesGenerator(
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
    # Default params
    _default_radius = 4
    _default_noise_freq = 0.1
    _default_noise_magn = 1
    _default_flatten_floor = True
    _default_fta_distance = 2
    # Random params
    _random_radius_interval = (1, 6)
    _random_noise_freq_interval = (0.3, 0)
    _random_noise_relative_magn_interval = (0.1, 0.7)
    _random_flatten_floor_probability = 0.5
    _random_fta_relative_distance_interval = [0, 1.5]

    @classmethod
    def from_defaults(cls):
        return TunnelPtClGenParams(
            radius=cls._default_radius,
            noise_freq=cls._default_noise_freq,
            noise_magn=cls._default_noise_magn,
            flatten_floor=cls._default_flatten_floor,
            fta_distance=cls._default_fta_distance,
        )

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_interval[0],
            cls._random_radius_interval[1],
        )
        noise_freq = np.random.uniform(
            cls._random_noise_freq_interval[0],
            cls._random_noise_freq_interval[1],
        )
        noise_magn = radius * np.random.uniform(
            cls._random_noise_relative_magn_interval[0],
            cls._random_noise_relative_magn_interval[1],
        )
        flatten_floor = cls._random_flatten_floor_probability > np.random.random()
        fta_distance = radius * np.random.uniform(
            cls._random_noise_relative_magn_interval[0],
            cls._random_noise_relative_magn_interval[1],
        )
        return TunnelPtClGenParams(
            radius=radius,
            noise_freq=noise_freq,
            noise_magn=noise_magn,
            flatten_floor=flatten_floor,
            fta_distance=fta_distance,
        )

    def __init__(
        self,
        radius=None,
        noise_freq=None,
        noise_magn=None,
        flatten_floor=None,
        fta_distance=None,
    ):
        assert not radius is None
        assert not noise_freq is None
        assert not noise_magn is None
        assert not flatten_floor is None
        assert not fta_distance is None
        self.radius = radius
        self.noise_freq = noise_freq
        self.flatter_floor = flatten_floor
        self.fta_distance = fta_distance


class IntersectionPtClGenParams:
    """Params that control how the pointcloud of an intersection is generated"""


class TunnelNetworkPtClGenParams:
    """Params that control the the overall generation of the pointcloud of the
    complete tunnel network"""


class TunnelNetworkMeshGenParams:
    """Params that control how the mesh is generated from the ptcl"""


class TunnelNewtorkMeshGenerator:
    def __init__(
        self,
        tunnel_network: TunnelNetwork,
        meshing_params: TunnelNetworkMeshGenParams,
    ):
        self._tunnel_network = tunnel_network
        self._ptcl_of_tunnel = dict()
        self._ptcl_of_intersections = dict()

    def _compute_tunnel_ptcl(
        self,
        tunnel: Tunnel,
        ptcl_gen_params: TunnelPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_intersection_ptcl(
        self,
        ptcl_gen_params: IntersectionPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_all_tunnels_ptcl(
        self,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_all_intersections_ptcl(self):
        raise NotImplementedError()

    def compute_all(
        self,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
        mesh_gen_params: TunnelNetworkMeshGenParams,
    ):
        self.compute_all_tunnel_ptcl(ptcl_gen_params)
        self.compute_all_intersection_ptcl(ptcl_gen_params)
        self.compute_mesh(mesh_gen_params)


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(tunnel: Tunnel, params: TunnelPtClGenParams):
    TODO
