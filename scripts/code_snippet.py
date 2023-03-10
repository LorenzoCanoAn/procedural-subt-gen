from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
import open3d as o3d

# PARAMS
mesh_save_path = "/home/lorenzo/mesh.obj"


def main():
    # Generate the grap
    tunnel_network = TunnelNetwork()
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    trial_limit = 200
    for _ in range(n_grown_tunnels):
        tunnel_network.add_random_tunnel_from_initial_node(trial_limit=trial_limit)
    for _ in range(n_connecting_tunnels):
        tunnel_network.add_random_tunnel_from_initial_to_final_node(
            trial_limit=trial_limit
        )
    tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
    tnm.clean_intersections()
    _, mesh = tnm.compute_mesh()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)


if __name__ == "__main__":
    main()
