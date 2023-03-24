from subt_proc_gen.mesh_generation import (
    CylindricalPerlinNoiseMapper,
    CylindricalPerlinNoiseMapperParms,
)
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import perf_counter_ns


def timeit(function, **args):
    start = perf_counter_ns()
    result = function(**args)
    end = perf_counter_ns()
    elapsed = (end - start) * 1e-9
    print(f"{function.__name__} took {elapsed:.5f} secs")
    return result


def generate_coords(scale):
    coords = [] * scale**2
    for i in range(scale):
        for j in range(scale):
            coords.append(np.array((i, j)))
    return coords


def noise_with_pool(generator: CylindricalPerlinNoiseMapper, coords):
    return generator(coords)


def fill_image_with_noise(image, noise, coords):
    for n, c in zip(noise, coords):
        image[c[0], c[1]] = n
    return image


def test1():
    scale = 1000
    generator = CylindricalPerlinNoiseMapper(
        scale, CylindricalPerlinNoiseMapperParms.from_defaults()
    )
    coords = timeit(generate_coords, scale=scale)
    noise = timeit(noise_with_pool, generator=generator, coords=coords)
    image = np.zeros((scale, scale))
    image = timeit(fill_image_with_noise, image=image, noise=noise, coords=coords)
    plt.imshow(image)
    plt.show()


def main():
    test1()


if __name__ == "__main__":
    main()
