# This script aims to ilustrate the procedural generation of tunnel network (the nodes are placed automatically)

import argparse
import os
import shutil
import pyvista as pv
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    IntersectionPtClType,
)
from subt_proc_gen.display_functions import plot_graph, plot_mesh, plot_splines
import numpy as np
import distinctipy
import logging as log

log.basicConfig(level=log.DEBUG)

####################################################################################################################################
# 	TOP LEVEL PARAMETERS
####################################################################################################################################
N_GROWN_TUNNELS = 3  # Pretty descriptive
N_CONNECTOR_TUNNELS = 2  # Pretty descriptive
FTA_DIST = (
    -1
)  # Distande from floor to axis. If negative, the floor of the tunnel is lower than the axis, for the time being, this applies
# to the complete tunnel network
mesh_save_path = "mesh.obj"


def main():
    plotter = pv.Plotter()
    # PREFACE

    # The main difference between this library and it's previous version is the use of Classes to store all the paramters that control
    # the random generation.
    # This script will be mainly explaining what each of these classes do, but there are some generalities:
    #   - A parameter class can be generated in three ways:
    #       1. Assigning the values of the attributes manually using the normal __init__ function
    #       2. Assigning the default values using the class method 'ParameterClass.from_defaults()-> ParamterClass'
    #       3. Assigning random values chosen automatically from a default range using the class method 'ParamterClass.random() -> ParameterClass'
    # All the parameter classes used in this script will be brefly explained, but the detailed explanation of each member of a class is given in the implementation

    # The procedural generation is divided in three steps:
    # 1. Creation of the tunnel network
    # 2. Creation of the pointcloud around the topological structure of the tunnel network
    # 3. Derivation of a mesh from the pointcloud

    ####################################################################################################################################
    # 	Generation of the tunnel network
    ####################################################################################################################################
    # To create an instance of a TunnelNetwork, it is necessary to initialise it with the corresponding paramter class. This parameter class
    # controlls very general aspects of the generation of the overal structure of the network, like the minimum distance from one intersection
    # to another, the maximum inclination that any tunnel can have, or if it is desired to have a completly flat tunnel network
    tunnel_network_params = TunnelNetworkParams.from_defaults()
    # Changed some values from defaults
    tunnel_network_params.min_distance_between_intersections = 30
    tunnel_network_params.collision_distance = 15
    # Create the tunnel network, by default, the class inits with a node already in the (0,0,0) position
    tunnel_network = TunnelNetwork(params=tunnel_network_params)
    for _ in range(N_GROWN_TUNNELS):
        params = GrownTunnelGenerationParams.random()
        # This function, internally, calls the class method 'Tunnel.grown()' with some extra checks
        tunnel_network.add_random_grown_tunnel(params=params, n_trials=100)
    for _ in range(N_CONNECTOR_TUNNELS):
        # This function, internally, calls the class method 'Tunnel.connector()' with some extra checks
        params = ConnectorTunnelGenerationParams.random()
        tunnel_network.add_random_connector_tunnel(n_trials=100)
    plot_graph(plotter, tunnel_network)
    plot_splines(plotter, tunnel_network, color="r")
    ####################################################################################################################################
    # 	Pointcloud and mesh generation
    ####################################################################################################################################
    ptcl_gen_params = TunnelNetworkPtClGenParams.random()
    mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_gen_params.fta_distance = FTA_DIST
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=ptcl_gen_params,
        meshing_params=mesh_gen_params,
    )
    mesh_generator.compute_all()
    plot_mesh(plotter, mesh_generator)
    plotter.show()
    if "y" in input("Save mesh (y/n):\n\t").lower():
        mesh_generator.save_mesh(mesh_save_path)


if __name__ == "__main__":
    main()
