from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import plot_nodes
import pyvista as pv


def test1():
    values = [0, 1, -1]
    tunnel_network = TunnelNetwork()
    for x in values:
        for y in values:
            for z in values:
                tunnel_network.add_node(Node((x, y, z)))

    plotter = pv.Plotter()
    plot_nodes(plotter, tunnel_network.nodes)
    plotter.show()


def main():
    test1()


if __name__ == "__main__":
    main()
