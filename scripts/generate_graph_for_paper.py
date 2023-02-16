import numpy as np
import matplotlib.pyplot as plt

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.tunnel import *
import random


def main():
    name = "MSL30NPN4"
    for i in range(1):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        # Generate the graph
        graph = TunnelNetwork()
        central_node = CaveNode()
        tunnel_params = TunnelParams(
            {
                "distance": 150,
                "starting_direction": angles_to_vector(
                    (np.deg2rad(-50), np.deg2rad(-5))
                ),
                "horizontal_tendency": np.deg2rad(-5),
                "horizontal_noise": np.deg2rad(0),
                "vertical_tendency": np.deg2rad(-5),
                "vertical_noise": np.deg2rad(0),
                "segment_length": 30,
                "segment_length_noise": 0,
                "node_position_noise": 10,
            }
        )
        print("first tunnel")
        tunnel_1 = Tunnel(graph, initial_node=central_node, params=tunnel_params)
        tunnel_params["starting_direction"] = angles_to_vector(
            (np.deg2rad(-10), np.deg2rad(10))
        )
        tunnel_params["horizontal_tendency"] = np.deg2rad(10)
        tunnel_params["vertical_tendency"] = np.deg2rad(5)
        print("second tunnel")
        tunnel_2 = Tunnel(graph, initial_node=central_node, params=tunnel_params)
        for i in range(1):
            i_node = tunnel_1.nodes[-2]
            f_node = tunnel_2.nodes[-2]
            assert isinstance(i_node, CaveNode)
            assert isinstance(f_node, CaveNode)
            print("last tunnel")
            tunnel_params["segment_length"] = 30
            tunnel_params["node_position_noise"] = 0
            last_tunnel = Tunnel(
                graph,
                initial_node=i_node,
                final_node=f_node,
                params=tunnel_params,
            )
            print(last_tunnel.success)
        debug_plot(graph, in_3d=True, wait="")
        # plt.autoscale(tight=True)
        plt.savefig(
            "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures/{}.svg".format(
                name
            ),
            bbox_inches="tight",
            pad_inches=0,
        )
        # with open("datafiles/graph.pkl", "wb") as f:
        #    pass
        # pickle.dump(graph, f)


if __name__ == "__main__":
    main()
