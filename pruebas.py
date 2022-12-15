import pyvista
import numpy as np
import os
spheres = []
filenames = os.listdir("meshes")
def key(n:str):
    return int(os.path.splitext(n)[0])
filenames.sort(key=key)
print(filenames)
COLORS = ["r", "g", "b", "c", "m", "y", "k", "tab:purple",
          "violet", "tomato", "teal", "steelblue", "springgreen"]


def fuse_two_meshes(mesh1, mesh2):
    mesh3 = mesh1.boolean_union(mesh2, progress_bar = True)
    return mesh3

# OPEN MESH FILES
for file in filenames:
    file_path = os.path.join("meshes", file)
    mesh = pyvista.read(file_path)
    mesh.flip_normals()
    mesh.smooth(n_iter=100,progress_bar=True, inplace=True)
    spheres.append(mesh)

# PLOT ALL MESHES

# for i in range(len(spheres)):
#     pl = pyvista.Plotter()
#     for n, sphere in enumerate(spheres[0:i+1]):
#         _ = pl.add_mesh(sphere, color=COLORS[n], style='wireframe', line_width=3)
#     pl.camera_position = 'xz'
#     pl.show()


for n, sphere in enumerate(spheres):
    if n == 0:
        fused_sphere = sphere
    else:
        fused_sphere = fuse_two_meshes(fused_sphere, sphere)

pl = pyvista.Plotter()
_ = pl.add_mesh(fused_sphere, color='tan')
for sphere in spheres:
    _ = pl.add_mesh(sphere, color='r', style='wireframe', line_width=3)
pl.camera_position = 'xz'
pl.show()
