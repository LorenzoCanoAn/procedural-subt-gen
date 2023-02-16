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
    for i in range(1):
        # Generate the graph
        fig = plt.figure(figsize=(10, 10))
        graph = TunnelNetwork()
        central_node = CaveNode()
        tunnel_params = TunnelParams(
            {
                "distance": np.random.uniform(50, 50),
                "starting_direction": np.array((1, 0, 0)),
                "horizontal_tendency": np.deg2rad(36),
                "horizontal_noise": np.deg2rad(0),
                "vertical_tendency": np.deg2rad(0),
                "vertical_noise": np.deg2rad(0),
                "segment_length": 10,
                "segment_length_noise": 0,
                "node_position_noise": 0,
            }
        )
        tunnel_0 = Tunnel(graph, initial_node=central_node, params=tunnel_params)
        n0 = tunnel_0.nodes[-2]
        n1 = tunnel_0.nodes[-1]
        dir = n1.xyz - n0.xyz
        dir /= np.linalg.norm(dir)
        th, ph = vector_to_angles(dir)
        th += np.deg2rad(36)
        dir = angles_to_vector((th, ph))
        tunnel_params["starting_direction"] = dir
        tunnel_1 = Tunnel(graph, initial_node=n1, params=tunnel_params)
        # Tunnel(graph, tunnel_0.nodes[0], tunnel_0.nodes[-1], params=tunnel_params)
        debug_plot(graph, in_3d=True)
        plt.show()
        with open("datafiles/graph.pkl", "wb") as f:
            pickle.dump(graph, f)

        # ds, ps, vs = tunnel.spline.discretize(0.05)
        # ds = np.reshape(ds, [-1, 1])
        # combined_spline_info = np.hstack([ds, ps, vs])
        # np.savetxt("datafiles/tunnel_info.txt", combined_spline_info)


if __name__ == "__main__":
    main()
