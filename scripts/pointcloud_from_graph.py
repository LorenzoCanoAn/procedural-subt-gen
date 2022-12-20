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
    plotter = pv.Plotter()
    for n, tunnel_with_mesh in enumerate(tunnel_network_with_mesh._tunnels_with_mesh):
        try:
            plotter.add_mesh(
                pv.PolyData(tunnel_with_mesh.central_points),
                color=COLORS[n],
            )
        except:
            pass
        plotter.add_mesh(
            pv.PolyData(
                tunnel_with_mesh.selected_end_points,
            ),
            color=COLORS[n],
        )
    plotter.show()
    exit()
    for intersection in tunnel_network.intersections:
        for n_tunnel_i, tunnel in enumerate(intersection.tunnels):
            # Plot the central points of the tunnel
            tunnel_with_mesh_i = TunnelWithMesh.tunnel_to_tunnelwithmesh(tunnel)

            for n_tunnel_j, tunnel_j in enumerate(intersection.tunnels):
                tunnel_with_mesh_j = TunnelWithMesh.tunnel_to_tunnelwithmesh(tunnel_j)
                if tunnel_with_mesh_i is tunnel_with_mesh_j:
                    continue
                # Update the secondary tunnel
                for point in tunnel_with_mesh_i.selected_points_of_end(intersection):
                    to_deselect = get_indices_of_points_below_cylinder(
                        tunnel_with_mesh_j.selected_points_of_end(intersection),
                        point,
                        0.3,
                    )
                    tunnel_with_mesh_j.deselect_point_of_end(intersection, to_deselect)

    # for n, tunnel_with_mesh in enumerate(tunnels_with_mesh):
    #     try:
    #         plotter.add_mesh(
    #             pv.PolyData(tunnel_with_mesh.central_points),
    #             color=COLORS[n],
    #         )
    #     except:
    #         pass
    #     plotter.add_mesh(
    #         pv.PolyData(
    #             tunnel_with_mesh.selected_end_points,
    #         ),
    #         color=COLORS[n],
    #     )

    # plotter.show()


if __name__ == "__main__":
    main()
