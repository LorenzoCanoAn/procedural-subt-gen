import pickle
from subt_proc_gen.mesh_generation import *
import pyvista as pv
from subt_proc_gen.PARAMS import *
from time import time_ns as ns
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork
from subt_proc_gen.helper_functions import gen_cylinder_around_point
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
        graph = pickle.load(f)
    # Order the tunnels so that the meshes intersect
    assert isinstance(graph, TunnelNetwork)

    tunnels_with_mesh = list()

    for n, tunnel in enumerate(graph.tunnels):
        print(f"Generating ptcl {n+1:>3} out of {len(graph.tunnels)}", end=" // ")
        start = ns()
        tunnel_with_mesh = TunnelWithMesh(tunnel)
        print(f"Time: {(ns()-start)*1e-9:<5.2f} s", end=" // ")
        print(f"{tunnel_with_mesh.n_points:<5} points")
        tunnels_with_mesh.append(tunnel_with_mesh)
    try:
        shutil.rmtree("screenshots")
    except:
        pass
    os.mkdir("screenshots")

    plotter = pv.Plotter(off_screen=True)
    for intersection in graph.intersections:
        threshold_distance = 5
        common_nodes = tunnel_i.tunnel.common_nodes(tunnel_j.tunnel)
        for common_node in common_nodes:
            plotter.add_mesh(pv.PolyData(tunnel_i.raw_points), color="red")
            plotter.add_mesh(pv.PolyData(tunnel_j.raw_points), color="blue")
            plotter.add_mesh(pv.PolyData(tunnel_i.points_in_ends(common_node)))
            plotter.add_mesh(pv.PolyData(tunnel_j.points_in_ends(common_node)))
            plotter.add_mesh(
                pv.PolyData(
                    gen_cylinder_around_point(
                        common_node.xyz, 20, threshold_distance
                    ),
                ),
                color="green",
            )
            plotter.camera_position = 'xy'
    plotter.show(screenshot=f"screenshots/i{i}_j{j}.png")


if __name__ == "__main__":
    main()
