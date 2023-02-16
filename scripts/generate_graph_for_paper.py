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
    name = "none"
    fig = plt.figure(figsize=(10, 10))
    for i in range(1):
        for i in range(1):
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            # Generate the graph
            graph = TunnelNetwork()
            central_node = CaveNode()
            tunnel_params = TunnelParams(
                {
                    "distance": 150,
                    "starting_direction": angles_to_vector(
                        (np.deg2rad(-30), np.deg2rad(-5))
                    ),
                    "horizontal_tendency": np.deg2rad(-5),
                    "horizontal_noise": np.deg2rad(0),
                    "vertical_tendency": np.deg2rad(-5),
                    "vertical_noise": np.deg2rad(0),
                    "segment_length": 30,
                    "segment_length_noise": 0,
                    "node_position_noise": 0,
                }
            )
            t1 = Tunnel(graph, initial_node=central_node, params=tunnel_params)
            tunnel_params["starting_direction"] = angles_to_vector(
                (np.deg2rad(30), np.deg2rad(-5))
            )
            t2 = Tunnel(graph, initial_node=central_node, params=tunnel_params)
            success = False
            while not success:
                tunnel_params["distance"] = 200
                tunnel_params["starting_direction"] = angles_to_vector(
                    (np.deg2rad(30), np.deg2rad(0))
                )
                print("Second tunnel")
                t_show = Tunnel(
                    graph,
                    initial_node=t1.nodes[-2],
                    params=tunnel_params,
                    override_checks=True,
                )
                success = t_show.success
            debug_plot(graph, wait="", clear=True)
            # Plot the points of the first tunnel
            p = t1.spline.discretize(5)[1]
            plt.scatter(
                x=p[:, 0],
                y=p[:, 1],
                alpha=1,
                s=50,
                c="k",
            )
            p = t2.spline.discretize(5)[1]
            plt.scatter(
                x=p[:, 0],
                y=p[:, 1],
                alpha=1,
                s=50,
                c="k",
            )
            p = t_show.get_points_to_check_collisions_with_other_tunnels(
                min_dist=MIN_DIST_OF_TUNNEL_COLLISSIONS, precision=5
            )
            plt.scatter(
                x=p[:, 0],
                y=p[:, 1],
                alpha=1,
                s=50,
                c="k",
            )

            plt.waitforbuttonpress()
            if False:
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
