import numpy as np
from perlin_numpy import generate_fractal_noise_2d, generate_fractal_noise_3d


class CylindricalPerlinNoiseGenerator:
    def __init__(
        self,
        length: float,
        radius: float,
        res: int,
        octaves: int = 3,
        persistence: float = 2,
        lacunarity: int = 2,
        min_precision_size: int = 100,
    ):
        self.lenght = length
        self.radius = radius
        self.circumference = self.radius * 2 * np.pi
        self.ratio_len_cirum = int(np.ceil(self.lenght / self.circumference))
        self.res = (int(res * self.ratio_len_cirum), int(res))
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.lacunarity = int(lacunarity)
        self.min_dimension = (self.lacunarity ** (self.octaves - 1)) * res
        if self.min_dimension < min_precision_size:
            self.min_dimension *= int(np.ceil(min_precision_size / self.min_dimension))
        self.shape = np.array(
            (
                self.min_dimension * self.ratio_len_cirum,
                self.min_dimension,
            ),
            dtype=int,
        )
        self.image = generate_fractal_noise_2d(
            shape=self.shape,
            res=self.res,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            tileable=(False, True),
        )

    def __call__(self, coords: np.ndarray):
        """Coords must be an array of size Nx2, the first column must be the position along the axis, and the second the angle around the cross section"""
        l_coord = coords[:, 0] / self.lenght * self.shape[0]
        a_coord = coords[:, 1] / (2 * np.pi) * self.shape[1]
        scaled_coords = np.rint(np.vstack([l_coord, a_coord]).T).astype(int)
        return self.image[scaled_coords]


class SphericalPerlinNoiseMapper:
    def __init__(
        self,
        radius: float,
        res: int,
        octaves: int = 3,
        persistence: float = 2,
        lacunarity: int = 2,
        min_precision_size: int = 100,
    ):
        self.res = (int(res), int(res), int(res))
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.lacunarity = int(lacunarity)
        self.min_dimension = (self.lacunarity ** (self.octaves - 1)) * res
        if self.min_dimension < min_precision_size:
            self.min_dimension *= int(np.ceil(min_precision_size / self.min_dimension))
        self.shape = np.array(
            (
                self.min_dimension,
                self.min_dimension,
                self.min_dimension,
            ),
            dtype=int,
        )
        self.image = generate_fractal_noise_3d(
            shape=self.shape,
            res=self.res,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            tileable=(False, False, False),
        )

    def __call__(self, coords: np.ndarray):
        """Coords must be an array of size Nx3, each row representing an xyz coordinate. Each coordinate can go from -1 to +1"""
        scaled_coords = np.rint((coords + 1) / 2 * self.min_dimension).astype(int)
        return self.image[scaled_coords]
