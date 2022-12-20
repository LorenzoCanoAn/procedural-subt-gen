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
import os


def tunnel_interesects_with_list(tunnel: Tunnel, list_of_tunnels):
    for tunnel_in_list in list_of_tunnels:
        assert isinstance(tunnel_in_list, Tunnel)
        if tunnel._nodes[-1] in tunnel_in_list._nodes:
            return True
        elif tunnel._nodes[0] in tunnel_in_list._nodes:
            return True
    return False


def main():
    # Generate the vertices of the mesh
    with open("graph.pkl", "rb") as f:
        tunnel_network = pickle.load(f)
    # Order the tunnels so that the meshes intersect
    assert isinstance(tunnel_network, TunnelNetwork)
    dist_threshold = 7
    tunnels_with_mesh = list()

    tunnel_network_with_mesh = TunnelNetworkWithMesh(tunnel_network)
    tunnel_network_with_mesh.clean_intersections()
    points, normals = tunnel_network_with_mesh.mesh_points_and_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
    o3d.visualization.draw_geometries(
        [mesh],
    )
    o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)


if __name__ == "__main__":
    main()
