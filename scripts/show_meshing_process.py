import numpy as np
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import TunnelParams, Tunnel, TunnelNetwork
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import *
from subt_proc_gen.PARAMS import N_ANGLES_PER_CIRCLE
from subt_proc_gen.mesh_generation import TunnelWithMesh, TunnelMeshingParams
import matplotlib
import pickle

matplotlib.rcParams.update({"font.size": 25})
import matplotlib.pyplot as plt
import pyvista as pv
import os

interactive = False
if interactive:
    window_size = 1000, 1000
    size_mult = 1
else:
    window_size = 3000, 3000
    size_mult = 3
save_file_name = "ptcl_gen_4.png"
save_folder = "/home/lorenzo/Documents/my_papers/IROS2023_proc/figures"
arrow_color = "r"
shpere_color = "b"
backgroudn_color = "white"
axis_point_radius = 0.1
plot_axis_related_things = True
plot_pointcloud_without_noise = True
noise = 0.3
flatten_floor = True


def main():
    save_path = os.path.join(save_folder, save_file_name)
    fig = plt.figure(figsize=(10, 10))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # Generate the graph
    graph = TunnelNetwork()
    central_node = CaveNode()
    tunnel_params = TunnelParams(
        {
            "distance": 30,
            "starting_direction": angles_to_vector((np.deg2rad(-30), np.deg2rad(0))),
            "horizontal_tendency": np.deg2rad(40),
            "horizontal_noise": np.deg2rad(0),
            "vertical_tendency": np.deg2rad(0),
            "vertical_noise": np.deg2rad(0),
            "segment_length": 10,
            "segment_length_noise": 0,
            "node_position_noise": 0,
        }
    )
    tunnel = Tunnel(graph, tunnel_params)
    tunnel.compute(central_node)
    axis_points = tunnel.spline.discretize(1)[1]
    # Create the arrows
    arrows = []
    for n_point in range(1, len(axis_points)):
        start = axis_points[n_point - 1]
        end = axis_points[n_point]
        direction = end - start
        arrows.append(pv.Arrow(start=start, direction=direction))
    # Create the spheres
    spheres = []
    for n_point, point in enumerate(axis_points):
        if n_point == len(axis_points) - 1:
            continue
        spheres.append(pv.Sphere(radius=axis_point_radius, center=point))

    plotter = pv.Plotter(off_screen=not interactive)
    plotter.window_size = window_size
    ####################################################################################################################################
    # 	PLOT THE AXIS-RELATED THINGS
    ####################################################################################################################################
    if plot_axis_related_things:
        # Plot the arrows
        for line in arrows:
            plotter.add_mesh(line, color=arrow_color)
        # Plot the spheres
        for sphere in spheres:
            plotter.add_mesh(sphere, color=shpere_color)
    ####################################################################################################################################
    # 	Plot the pointcloud
    ####################################################################################################################################
    if plot_pointcloud_without_noise:
        params = TunnelMeshingParams(
            {
                "roughness": noise,
                "flatten_floor": flatten_floor,
                "floor_to_axis_distance": 1,
                "radius": 2,
            }
        )
        twm = TunnelWithMesh(tunnel, meshing_params=params)
        pointcloud = pv.PolyData(twm.all_selected_points)
        scalars = np.ones(len(twm.all_selected_points))
        if not flatten_floor:
            for n_circle in range(len(axis_points)):
                if n_circle == 0:
                    color = 0
                else:
                    color = 10
                for n_p in range(N_ANGLES_PER_CIRCLE):
                    scalars[n_circle * N_ANGLES_PER_CIRCLE + n_p] = color
                    if n_circle == 0:
                        line = pv.Line(
                            pointa=axis_points[n_circle],
                            pointb=twm.all_selected_points[
                                n_circle * N_ANGLES_PER_CIRCLE + n_p
                            ],
                            resolution=3,
                        )
                        line_color = "g" if n_circle == 0 else "gray"
                        plotter.add_mesh(
                            line, color=line_color, line_width=3 * size_mult
                        )
        else:
            n1 = 33
            n2 = 957
            scalars = np.ones(n1) * 0
            scalars = np.concatenate([scalars, np.ones(n2) * 10])
            scalars = np.concatenate([scalars, np.ones(1500 - n1 - n2) * 2])
        plotter.add_mesh(
            pointcloud,
            scalars=scalars,
            show_scalar_bar=False,
            cmap="Dark2",
            render_points_as_spheres=True,
            point_size=15 * size_mult,
        )
        plotter.camera.position = (
            -10.394597091978623,
            3.115605826202874,
            6.632915464573019,
        )
    plotter.camera.focal_point = (
        10.080409210990878,
        -1.1264967608208014,
        -3.620568643196208,
    )
    plotter.camera.roll = 98.57392487258856
    _ = plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=1,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )

    if interactive:
        plotter.show()
        print("plotter.camera.position =", plotter.camera.position)
        print("plotter.camera.focal_point =", plotter.camera.focal_point)
        print("plotter.camera.roll =", plotter.camera.roll)
    else:
        plotter.show(screenshot=save_path, window_size=window_size)


if __name__ == "__main__":
    main()
