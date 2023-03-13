from subt_proc_gen.tunnel import (
    TunnelNetwork,
    Tunnel,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.graph import Node
from subt_proc_gen.geometry import Vector3D
from subt_proc_gen.display_functions import plot_nodes, plot_edges, plot_xyz_axis
import numpy as np
import pyvista as pv
import os
import pathlib


def test1():
    values = [0, 1, -1]
    tunnel_network = TunnelNetwork()
    for x in values:
        for y in values:
            for z in values:
                tunnel_network.add_node(Node((x, y, z)))


def test2():
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    tunnel_network.add_tunnel(
        Tunnel.grown(
            i_node=first_node, params=GrownTunnelGenerationParams.from_defaults()
        )
    )
    params = GrownTunnelGenerationParams.random()
    tunnel_network.add_tunnel(Tunnel.grown(i_node=first_node, params=params))


def test3():
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        params=GrownTunnelGenerationParams.from_defaults(
            initial_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(0), length=30
            )
        ),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        params=GrownTunnelGenerationParams.from_defaults(
            initial_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(90), length=30
            )
        ),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        i_node=first_tunnel[-2],
        f_node=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.remove_tunnel(first_tunnel)
    tunnel_network.add_tunnel(third_tunnel)


def test4():
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        params=GrownTunnelGenerationParams.from_defaults(
            initial_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(0), length=30
            )
        ),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        params=GrownTunnelGenerationParams.from_defaults(
            initial_direction=Vector3D.from_inclination_yaw_length(
                inclination=0, yaw=np.deg2rad(90), length=30
            )
        ),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        i_node=first_tunnel[-2],
        f_node=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(third_tunnel)
    tunnel_network.to_yaml(os.path.join(pathlib.Path.home(), "test.yaml"))
    del tunnel_network
    tunnel_network = TunnelNetwork()
    tunnel_network.load_yaml(os.path.join(pathlib.Path.home(), "test.yaml"))
    plotter = pv.Plotter()
    plot_nodes(plotter, tunnel_network.nodes, color="r")
    plot_edges(plotter, tunnel_network.edges, color="b")
    plot_xyz_axis(plotter)
    plotter.show()


def main():
    test1()
    test2()
    test3()
    test4()


if __name__ == "__main__":
    main()
