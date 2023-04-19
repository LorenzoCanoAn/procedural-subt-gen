from enum import Enum
import numpy as np
import time
from perlin_noise import PerlinNoise
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
        scaled_coords[np.where(scaled_coords > 1.0)] = 1.0
        scaled_coords[np.where(scaled_coords < 0.0)] = 0.0
        pool = Pool()
        noises = pool.map(self._noise_of_scaled_coords, scaled_coords)
        pool.close()
        return np.array(noises)

    def call_no_multiprocessing(self, coords):
        if type(coords) in [tuple, list]:
            coords = np.array(coords)
        scaled_coords = coords / self._sampling_scale
        return np.array(
            [
                self._noise_of_scaled_coords(scaled_coord)
                for scaled_coord in scaled_coords
            ]
        )


class SphericalPerlinNoiseMapperParms:
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


class SphericalPerlinNoiseMapper:
    def __init__(
        self, sampling_scale, params: SphericalPerlinNoiseMapperParms, seed=None
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
        pool = Pool()
        noises = pool.map(self._noise_of_scaled_coords, coords)
        pool.close()
        return np.array(noises)

    def call_no_multiprocessing(self, coords):
        if type(coords) in [tuple, list]:
            coords = np.array(coords)
        scaled_coords = coords / self._sampling_scale
        return np.array(
            [
                self._noise_of_scaled_coords(scaled_coord)
                for scaled_coord in scaled_coords
            ]
        )
