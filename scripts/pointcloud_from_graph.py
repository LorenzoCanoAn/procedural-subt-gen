import pickle
from subt_proc_gen.mesh_generation import *
import pyvista as pv
from subt_proc_gen.PARAMS import *
from time import time_ns as ns
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork

MESH_FOLDER = "meshes"
import matplotlib.pyplot as plt


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

    for tunnel_i in tunnels_with_mesh:
        for tunnel_j in tunnels_with_mesh:
            if not tunnel_i is tunnel_j:
                assert isinstance(tunnel_i, TunnelWithMesh) and isinstance(
                    tunnel_i, TunnelWithMesh
                )
                threshold_distance = 5
                common_nodes = tunnel_i.tunnel.common_nodes(tunnel_j.tunnel)
                for common_node in common_nodes:
                    points_i = tunnel_i.raw_points.T
                    points_j = tunnel_j.points.T

                    points_i_xy = tunnel_i.raw_points.T[:, :2]
                    points_j_xy = tunnel_j.points.T[:, :2]
                    differences_i = points_i_xy - np.reshape(
                        common_node.xyz[:2], [1, -1]
                    )
                    differences_j = points_j_xy - np.reshape(
                        common_node.xyz[:2], [1, -1]
                    )
                    distances_i = np.linalg.norm(differences_i, axis=1)
                    distances_j = np.linalg.norm(differences_j, axis=1)
                    indices_i = np.where(distances_i < threshold_distance)
                    indices_j = np.where(distances_j < threshold_distance)

                    selected_i = points_i[indices_i]
                    selected_j = points_j[indices_j]

                    plotter = pv.Plotter()
                    plotter.add_mesh(pv.PolyData(points_i), color="red")
                    plotter.add_mesh(pv.PolyData(points_j), color="blue")

                    plotter.add_mesh(pv.PolyData(selected_i))
                    plotter.add_mesh(pv.PolyData(selected_j))

                    for i in range(1000):
                        angle = np.random.uniform(0, 2 * np.pi)
                        x = np.cos(angle) * threshold_distance
                        y = np.sin(angle) * threshold_distance
                        z = np.random.uniform(-10, 10)
                        vector = np.array((x, y, z))
                        plotter.add_mesh(
                            pv.PolyData(common_node.xyz + vector), color="green"
                        )
                    plotter.show()


if __name__ == "__main__":
    main()
