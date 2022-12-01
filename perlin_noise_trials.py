import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
noise = PerlinNoise(octaves=100, seed=1)
xpix, ypix = 1000, 1000
pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]

plt.imshow(pic, cmap='gray')
plt.show()
