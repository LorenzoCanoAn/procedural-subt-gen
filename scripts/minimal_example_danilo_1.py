# This script illustrates the most manual way of generating a tunnel network, meaning that the nodes are "hard-coded"
import random
import sys

# sys.path.append("../src")
import argparse
import os
import shutil
import pyvista as pv
from subt_proc_gen.tunnel import (
    Tunnel,
    TunnelNetwork,
    TunnelNetworkParams,
    TunnelType,
    Node,
    ConnectorTunnelGenerationParams,
    GrownTunnelGenerationParams,
)

from subt_proc_gen.param_classes import (
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
)

from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    IntersectionPtClType,
    TunnelPtClGenParams,
)
from subt_proc_gen.display_functions import (
    plot_graph,
    plot_splines,
    plot_mesh,
    plot_intersection_ptcls,
    plot_tunnel_ptcls,
    plot_ptcl,
)
import numpy as np
import distinctipy
import logging as log

log.basicConfig(level=log.DEBUG)

####################################################################################################################################
# 	TOP LEVEL PARAMETERS
####################################################################################################################################
FTA_DIST = (
    -1
)  # Distande from floor to axis. If negative, the floor of the tunnel is lower than the axis, for the time being, this applies
# to the complete tunnel network
mesh_save_path = "mesh.obj"


def main():
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
    tunnel_network = TunnelNetwork(params=tunnel_network_params, initial_node=False)
    # At its core, a Tunnel is just a collection of nodes, to create a tunnel manually from nodes:
    nodes_of_tunnel_1 = [
        Node((0, 0, 0)),  # The first is the default at (0,0,0)
        Node((20, 0, 0)),  # Nodes are only initialized by their coordinates
        Node((40, 0, 10)),
        Node((80, 0, 0)),
    ]
    tunnel_1 = Tunnel()
    tunnel_1.append_node(Node(0, 0, 0))
    tunnel_1.append_node(Node(20, 0, 0))
    tunnel_1.append_node(Node(40, 0, 10))
    tunnel_1.append_node(Node(80, 0, 0))

    tunnel_2 = Tunnel()
    tunnel_2.append_node(tunnel_1[2])
    tunnel_2.append_node(Node((40, 50, -40)))
    tunnel_2.append_node(Node((0, 60, -80)))

    # Tunnels are just a list of nodes

    # You can generate tunnels randomly, but this method is not recomended as it does not do any checks,
    # in the second script there is the recomended method
    # example_tunnel = Tunnel.grown(
    #    nodes_of_tunnel_1[1], (0, -1, 0), GrownTunnelGenerationParams.random()
    # )
    # Then the tunnels are added to the network, and everyting is taken care of (intersections and other thingies)
    # DO NOT ADD NODES DIRECTLY TO THE TUNNEL NETWORK, ONLY ADD TUNNELS
    tunnel_network.add_tunnel(tunnel_1)
    tunnel_network.add_tunnel(tunnel_2)
    # PLOT DE LAS SPLINE Y GRAFO
    plotter = pv.Plotter()
    plot_graph(plotter, tunnel_network)
    plot_splines(plotter, tunnel_network, color="r")
    plotter.show()
    ####################################################################################################################################
    # 	Pointcloud and mesh generation
    ####################################################################################################################################
    np.random.seed(0)
    ptcl_gen_params = TunnelNetworkPtClGenParams.random()
    # To add specific ptcl generation parameters for a specific tunnel:
    tunnel_1_ptcl_gen_params = TunnelPtClGenParams.from_defaults()
    tunnel_1_ptcl_gen_params.radius = 7
    ptcl_gen_params.pre_set_tunnel_params[tunnel_1] = tunnel_1_ptcl_gen_params
    mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_gen_params.fta_distance = FTA_DIST
    mesh_generator = TunnelNetworkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=ptcl_gen_params,
        meshing_params=mesh_gen_params,
    )
    mesh_generator.compute_all()
    # PLOT DE LA POINTCLOUD
    plotter = pv.Plotter()
    plot_intersection_ptcls(plotter, mesh_generator, color="r")
    plot_tunnel_ptcls(plotter, mesh_generator, color="b")
    # PLOT DE LA MESH
    plot_mesh(plotter, mesh_generator)
    plotter.show()
    # SACAR INFOR DE LAS SPLINES
    if "y" in input("Save mesh (y/n):\n\t").lower():
        mesh_generator.save_mesh(mesh_save_path)


if __name__ == "__main__":
    main()
