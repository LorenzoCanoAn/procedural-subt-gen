from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
)
from subt_proc_gen.gazebo_file_gen import mesh_generator_to_gazebo_model
import logging as log

log.basicConfig(level=log.DEBUG)


def main():
    tunnel_network = TunnelNetwork()
    tunnel_network.add_random_grown_tunnel()
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network, TunnelNetworkPtClGenParams()
    )
    mesh_generator.compute_all()
    mesh_generator_to_gazebo_model(mesh_generator, "/home/lorenzo/model")


if __name__ == "__main__":
    main()
