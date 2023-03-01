from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh, TunnelMeshingParams
import pickle

meshing_params = [
    TunnelMeshingParams(
        params={
            "roughness": 0,
            "flatten_floor": False,
            "floor_to_axis_distance": 2,
            "radius": 3,
        }
    ),
    TunnelMeshingParams(
        params={
            "roughness": 0.4,
            "flatten_floor": False,
            "floor_to_axis_distance": 2,
            "radius": 3,
        }
    ),
    TunnelMeshingParams(
        params={
            "roughness": 0.1,
            "flatten_floor": True,
            "floor_to_axis_distance": 2,
            "radius": 3,
        }
    ),
    TunnelMeshingParams(
        params={
            "roughness": 0,
            "flatten_floor": True,
            "floor_to_axis_distance": 1,
            "radius": 4,
        }
    ),
]


def main():
    # Generate the graph
    for i in range(10):
        tunel_network = TunnelNetwork()
        for _ in range(3):
            tunel_network.add_random_tunnel_from_initial_node(trial_limit=200000)
        for _ in range(2):
            tunel_network.add_random_tunnel_from_initial_to_final_node(
                trial_limit=200000
            )
        for j in range(4):
            tunel_network_with_mesh = TunnelNetworkWithMesh(
                tunnel_network=tunel_network, i_meshing_params=meshing_params[j]
            )
            tunel_network_with_mesh.clean_intersections()
            tunel_network_with_mesh.compute_mesh()
            tunel_network_with_mesh.save_mesh(
                f"datafiles/same_network_different_mesh_{i}_{j}.obj"
            )


if __name__ == "__main__":
    main()
