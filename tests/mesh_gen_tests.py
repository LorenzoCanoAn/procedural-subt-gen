from subt_proc_gen.mesh_generation import (
    CylindricalPerlinNoiseMapper,
    CylindricalPerlinNoiseMapperParms,
)
import numpy as np
import matplotlib.pyplot as plt


def test1():
    scalea = 400
    a = CylindricalPerlinNoiseMapper(
        scalea, CylindricalPerlinNoiseMapperParms.from_defaults()
    )
    img_shapea = (scalea, scalea)
    imga = np.zeros(img_shapea)
    for i in range(img_shapea[0]):
        for j in range(img_shapea[1]):
            coords = np.array((i, j))
            imga[i, j] = a(coords)
    plt.imshow(imga)
    plt.show()


def main():
    test1()


if __name__ == "__main__":
    main()
