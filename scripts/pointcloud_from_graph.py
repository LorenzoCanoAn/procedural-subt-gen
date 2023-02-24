import sys

sys.path.insert(0, "src")
import sys

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import pickle
from subt_proc_gen.mesh_generation import *
import pyvista as pv
from subt_proc_gen.PARAMS import *
from time import time_ns as ns
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork
from subt_proc_gen.helper_functions import (
    gen_cylinder_around_point,
    get_indices_of_points_below_cylinder,
)
import open3d as o3d
import shutil

MESH_FOLDER = "meshes"
import matplotlib.pyplot as plt
import pyvista as pv


def tunnel_interesects_with_list(tunnel: Tunnel, list_of_tunnels):
    for tunnel_in_list in list_of_tunnels:
        assert isinstance(tunnel_in_list, Tunnel)
        if tunnel._nodes[-1] in tunnel_in_list._nodes:
            return True
        elif tunnel._nodes[0] in tunnel_in_list._nodes:
            return True
    return False


def pc_from_graph(plotter, roughness, tunnel_network=None, filename=None, radius=5):
#    fig = plt.figure(figsize=(10, 10))
    if not tunnel_network:
        # Generate the vertices of the mesh
        with open("datafiles/graph.pkl", "rb") as f:
            tunnel_network = pickle.load(f)

    # Order the tunnels so that the meshes intersect
    assert isinstance(tunnel_network, TunnelNetwork)
    print("Looping oover tunnel network")
    if True:
        tunnel_network_with_mesh = TunnelNetworkWithMesh(
            tunnel_network,
            meshing_params=TunnelMeshingParams(
                {"roughness": roughness, "radius": radius, "floor_to_axis_distance": radius/4}
            ),
        )
        tunnel_network_with_mesh.clean_intersections()
        points, normals = tunnel_network_with_mesh.mesh_points_and_normals()
        # plotter = pv.Plotter()
        for i, mesh in enumerate(tunnel_network_with_mesh._tunnels_with_mesh):
            print(i)
            plotter.add_mesh(pv.PolyData(mesh.all_selected_points), color=COLORS[i])
        # plotter.show()
        for i, tunnel in enumerate(tunnel_network_with_mesh._tunnels_with_mesh):
            if i == 0:
                proj_points, proj_normals = tunnel.get_xy_projection(0.1)
            else:
                _proj_points, _proj_normals = tunnel.get_xy_projection(0.1)
                proj_points = np.vstack([proj_points, _proj_points])
                proj_normals = np.vstack([proj_normals, proj_normals])

            #plt.scatter(x=points[:, 0], y=points[:, 1], c="b")
        #plt.show()
        np.save("points", points)
        np.save("normals", normals)
    else:
        points = np.load("points.npy")
        normals = np.load("normals.npy")
    methods = [
        "poisson",
    ]
    if not filename is None:
        for method in methods:
            for poisson_depth in [11]:
                mesh, ptcl = mesh_from_vertices(
                    points, normals, method=method, poisson_depth=poisson_depth
                )
                # o3d.visualization.draw_geometries([ptcl])
                simplified_mesh = mesh.simplify_quadric_decimation(
                    int(len(mesh.triangles) * 0.3)
                )
                print(f"Original mesh has {len(mesh.triangles)}")
                print(f"Simplified mesh has {len(simplified_mesh.triangles)}")
                o3d.io.write_triangle_mesh(
                    filename,
                    # f"datafiles/{method}_depth_{poisson_depth}_simplified.obj",
                    simplified_mesh,
                )
    return proj_points, proj_normals


'''
class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(lambda : maino(self.plotter))
        meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()
        #maino(self.plotter)

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()

'''

if __name__ == "__main__":
    plotter = pv.Plotter()
    pc_from_graph(plotter, 0.00001)
    plotter.show()
