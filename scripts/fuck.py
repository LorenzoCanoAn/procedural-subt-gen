from subt_proc_gen.tunnel import Tunnel, TunnelNetwork, Node, GrownTunnelGenerationParams, ConnectorTunnelGenerationParams
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
import numpy as np
import random
from networkx import Graph, has_path, is_connected
from subt_proc_gen.param_classes import TunnelPtClGenParams
from pyvista.plotting.plotting import Plotter
import pyvista as pv
from subt_proc_gen.display_functions import plot_graph, plot_node, plot_mesh
from argparse import ArgumentParser
import logging
import os

logging.basicConfig(level=logging.DEBUG)

BASE_FOLDER = "/home/lorenzo/gazebo_worlds/topo_exp_tunnels"
ENV_NAME = "env_002"


def gen_text(text):
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
    <model name="tunnel_network">
        <static>true</static>
        <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{text}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>{text}</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def create_tn():
    n1 = Node(0, 0, 0)
    n2 = Node(35, 0, 0)
    n3 = Node(45, 20, 0)
    n4 = Node(30, -65, 0)
    n5 = Node(0, -60, 0)
    n6 = Node(80, -5, 0)
    n7 = Node(90, 30, 0)
    n8 = Node(100, -5, 0)
    n9 = Node(120, -6, 0)
    n10 = Node(110, 20, 0)
    n11 = Node(140, -20, 0)
    n12 = Node(70, -70, 0)
    ConnectorTunnelGenerationParams._random_node_position_horizontal_noise_interval = (0, 0)
    ConnectorTunnelGenerationParams._random_node_position_vertical_noise_interval = (0, 0)
    ConnectorTunnelGenerationParams._default_node_position_horizontal_noise = 0
    ConnectorTunnelGenerationParams._default_node_position_vertical_noise = 0
    t1 = Tunnel.connector(n1, n2)
    t2 = Tunnel.connector(n2, n3)
    t3 = Tunnel.connector(n2, n4)
    t4 = Tunnel.connector(n4, n5)
    t5 = Tunnel.connector(n4, n12)
    t6 = Tunnel.connector(n2, n6)
    t7 = Tunnel.connector(n6, n7)
    t8 = Tunnel.connector(n6, n8)
    t9 = Tunnel.connector(n8, n9)
    t10 = Tunnel.connector(n9, n10)
    t11 = Tunnel.connector(n9, n11)

    tn = TunnelNetwork()
    tn.add_tunnel(t1)
    tn.add_tunnel(t2)
    tn.add_tunnel(t3)
    tn.add_tunnel(t4)
    tn.add_tunnel(t5)
    tn.add_tunnel(t6)
    tn.add_tunnel(t7)
    tn.add_tunnel(t8)
    tn.add_tunnel(t9)
    tn.add_tunnel(t10)
    tn.add_tunnel(t11)
    return tn


def main(n):
    tn = create_tn()
    TunnelPtClGenParams._random_radius_interval = (6, 7)
    tnmg = TunnelNetworkMeshGenerator(tn)
    tnmg.compute_all(compute_floors=True)
    pv_mesh = tnmg.pyvista_mesh
    pv.plot(pv_mesh)

    path_to_env = os.path.join(BASE_FOLDER, ENV_NAME)
    os.makedirs(path_to_env, exist_ok=True)
    path_to_mesh = os.path.join(path_to_env, "mesh.obj")
    path_to_sdf = os.path.join(path_to_env, "model.sdf")
    sdf_content = gen_text(path_to_mesh)
    pv.save_meshio(path_to_mesh, pv_mesh)
    with open(path_to_sdf, "w+") as f:
        f.write(sdf_content)


if __name__ == "__main__":
    main(10)
