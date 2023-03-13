from subt_proc_gen.tunnel_2 import TunnelNetwork, Tunnel, GrownTunnelGenerationParams
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import plot_nodes, plot_edges, plot_xyz_axis
import pyvista as pv


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
    print(params)
    tunnel_network._tunnels
    plotter = pv.Plotter()
    plot_nodes(plotter, tunnel_network.nodes)
    plot_edges(plotter, tunnel_network.edges)
    plot_xyz_axis(plotter)
    plotter.show()


def main():
    test1()
    test2()


if __name__ == "__main__":
    main()
