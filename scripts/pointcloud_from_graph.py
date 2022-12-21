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
    with open("datafiles/graph.pkl", "rb") as f:
        tunnel_network = pickle.load(f)
    # Order the tunnels so that the meshes intersect
    assert isinstance(tunnel_network, TunnelNetwork)
    dist_threshold = 7
    tunnels_with_mesh = list()

    tunnel_network_with_mesh = TunnelNetworkWithMesh(
        tunnel_network, meshing_params=TunnelMeshingParams({"roughness": 0.5})
    )
    tunnel_network_with_mesh.clean_intersections()
    points, normals = tunnel_network_with_mesh.mesh_points_and_normals()

    mesh, ptcl = mesh_from_vertices(points, normals)

    simplified_mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * 0.3))
    print(f"Original mesh has {len(mesh.triangles)}")
    print(f"Simplified mesh has {len(simplified_mesh.triangles)}")
    o3d.io.write_triangle_mesh("datafiles/originial.ply", mesh)
    o3d.io.write_triangle_mesh("datafiles/simplified.ply", simplified_mesh)


if __name__ == "__main__":
    main()
