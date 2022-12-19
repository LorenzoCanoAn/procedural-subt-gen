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
                assert isinstance(tunnel_i, TunnelWithMesh) and isinstance(tunnel_i, TunnelWithMesh)
                threshold_distance = 5
                points_2, node = tunnel_i.get_points_to_delete_from_other_tunnel(tunnel_j, threshold_distance=threshold_distance)
                if not points_2 is None:
                    plotter = pv.Plotter()
                    plotter.add_mesh(pv.PolyData(tunnel_i.points.T),color="red")
                    plotter.add_mesh(pv.PolyData(tunnel_j.points.T),color="blue")
                    plotter.add_mesh(pv.PolyData(node.xyz),color="green")
                    for i in range(1000):
                        vector = np.random.uniform(-1,1,3)
                        vector /= np.linalg.norm(vector)
                        vector *= threshold_distance
                        plotter.add_mesh(pv.PolyData(node.xyz+vector),color="green")
                    print(points_2)
                    print(tunnel_i.points[:,points_2].T)
                    ##plotter.add_mesh(pv.PolyData(tunnel_i.points[:,points_2].T))
                    plotter.show()

if __name__ == "__main__":
    main()
