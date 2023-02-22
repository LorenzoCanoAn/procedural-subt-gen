import numpy as np

from subt_proc_gen.helper_functions import *
import pickle
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.display_functions import debug_plot, plot_spline_2d
from subt_proc_gen.tunnel import *
import random
from subt_proc_gen.helper_functions import what_points_are_close
import matplotlib

matplotlib.rcParams.update({"font.size": 25})
import matplotlib.pyplot as plt

spline_color = "b"
distance_between_spline_points = 7
color_of_close_points = "r"


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
            t1 = Tunnel(graph, params=tunnel_params)
            t1.compute(initial_node=central_node)
            tunnel_params["starting_direction"] = angles_to_vector(
                (np.deg2rad(20), np.deg2rad(-5))
            )
            t2 = Tunnel(graph, params=tunnel_params)
            t2.compute(initial_node=central_node)
            success = False
            while not success:
                tunnel_params["distance"] = 150
                tunnel_params["starting_direction"] = angles_to_vector(
                    (np.deg2rad(75), np.deg2rad(0))
                )
                print("Second tunnel")
                t_show = Tunnel(
                    graph,
                    params=tunnel_params,
                )
                t_show.compute(
                    initial_node=t1.nodes[-2],
                    do_checks=False,
                )
                success = t_show.success
            # Plot the splines
            for tunnel in graph.tunnels:
                plot_spline_2d(tunnel.spline, color=spline_color)
            # Plot the nodes of the graph
            for node in graph.nodes:
                x, y, z = node.xyz
                plt.scatter(
                    x=x,
                    y=y,
                    s=500,
                    zorder=-1,
                    edgecolors="k",
                    facecolors="none",
                    linewidths=3,
                )
            p1 = t1.spline.discretize(distance_between_spline_points)[1]
            plt.scatter(x=p1[:, 0], y=p1[:, 1], alpha=1, s=100, c="k", zorder=0)
            p2 = t2.spline.discretize(distance_between_spline_points)[1]
            plt.scatter(x=p2[:, 0], y=p2[:, 1], alpha=1, s=100, c="k", zorder=0)
            p3 = t_show.relevant_points_for_collision(
                distance_between_points=distance_between_spline_points,
                distance_to_intersection=MIN_DIST_OF_TUNNEL_COLLISSIONS * 1.5,
            )
            plt.scatter(x=p3[:, 0], y=p3[:, 1], alpha=1, s=100, c="k", zorder=0)
            ids2, ids3 = what_points_are_close(p2, p3, min_dist=10)
            plt.scatter(
                x=p3[ids3, 0],
                y=p3[ids3, 1],
                alpha=1,
                s=100,
                c=color_of_close_points,
                zorder=0,
            )
            plt.scatter(
                x=p2[ids2, 0],
                y=p2[ids2, 1],
                alpha=1,
                s=100,
                c=color_of_close_points,
                zorder=0,
            )
            axis_width = 200
            min_x = -10
            min_y = -axis_width / 2
            plt.gca().set_xlim(min_x, min_x + axis_width)
            plt.gca().set_ylim(min_y, min_y + axis_width)
            plt.show()
            if False:
                plt.savefig(
                    "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures/{}.svg".format(
                        name
                    ),
                    bbox_inches="tight",
                    pad_inches=0,
                )
            with open("datafiles/graph.pkl", "wb") as f:
                pickle.dump(graph, f)


if __name__ == "__main__":
    main()
