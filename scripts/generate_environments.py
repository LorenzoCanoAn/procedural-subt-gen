import argparse
import os
import shutil
import pyvista as pv
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    IntersectionPtClType,
)
import numpy as np
import distinctipy
import logging as log

colors = distinctipy.get_colors(30)

log.basicConfig(level=log.DEBUG)

MODEL_SDF_TEXT = """<?xml version="1.0"?>
<sdf version="1.6">
    <model name="tunnel_network">
        <static>true</static>
        <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def gen_axis_points_file(mesh_generator: TunnelNetworkMeshGenerator):
    axis_points = np.zeros((0, 3 + 3 + 1 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.ptcl_params_of_tunnel(tunnel).radius
        aps = mesh_generator.aps_of_tunnels
        avs = mesh_generator.avs_of_tunnels
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        tunnel_flags = np.ones((len(aps), 1)) * 1
        axis_points = np.concatenate(
            (axis_points, np.concatenate((aps, avs, rds, tunnel_flags), axis=1)), axis=0
        )
    for intersection in mesh_generator._tunnel_network.intersections:
        params = mesh_generator.params_of_intersection(intersection)
        if params.ptcl_type == IntersectionPtClType.spherical_cavity:
            radius = params.radius
        elif params.ptcl_type == IntersectionPtClType.no_cavity:
            radiuses = []
            for tunnel in mesh_generator._tunnel_network._tunnels_of_node[intersection]:
                radiuses.append(mesh_generator.ptcl_params_of_tunnel(tunnel).radius)
            radius = max(radiuses)
        else:
            raise NotImplementedError()
        aps = mesh_generator.aps_of_intersection(intersection)
        avs = mesh_generator.avs_of_intersection(intersection)
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        intersection_flags = np.ones((len(aps), 1)) * 2
        axis_points_of_inter = np.concatenate(
            (aps, avs, rds, intersection_flags), axis=1
        )
        axis_points = np.concatenate((axis_points, axis_points_of_inter), axis=0)
    return axis_points


def args():
    parser = argparse.ArgumentParser(
        prog="EnvironemtnsGenerator",
        description="This script generates as many environemts as desired, with different topological structures, and saves them in a folder.",
    )
    parser.add_argument("-F", "--folder", type=str, required=True)
    parser.add_argument(
        "-N",
        "--number_of_environments",
        type=int,
        required=True,
        help="Number of environemts that will be generated",
    )
    parser.add_argument(
        "-NGT",
        "--number_of_grown_tunnels",
        default=5,
        required=False,
        type=int,
        help="Number of tunnels that are generated from just an initial node, growing randomly from it.",
    )
    parser.add_argument(
        "-NCT",
        "--number_of_connector_tunnels",
        default=2,
        required=False,
        type=int,
        help="Number of tunnels that are generated from an initial node to a final node already present in the Tunnel Network",
    )
    parser.add_argument(
        "-_FTA",
        "--min_fta_distance",
        default=-2,
        required=False,
        help="This parameter controls what is the minimum vertical distance from the axis of a tunnel to the floor of the tunnel. If this number is positive, the floor of the tunnel will allways be avobe the axis. This number has to be lower thant -FTA",
    )
    parser.add_argument(
        "-FTA",
        "--max_fta_distance",
        default=-0.5,
        required=False,
        help="This parameter controls what is the maximum vertical distance from the axis of a tunnel to the floor of the tunnel. If this number is negative, the floor of the tunnel will allways be below the axis, This number has to be greater than -_FTA",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        required=False,
        default=False,
        type=bool,
        help="If this is set to True, the environmets previously on the folder will be overwriten",
    )
    return parser.parse_args()


def main():
    arguments = args()
    base_folder = arguments.folder
    n_envs = arguments.number_of_environments
    n_grown = arguments.number_of_grown_tunnels
    n_connector = arguments.number_of_connector_tunnels
    overwrite = arguments.overwrite
    min_fta_distance = arguments.min_fta_distance
    max_fta_distance = arguments.max_fta_distance
    fta_dist = np.random.uniform(min_fta_distance, max_fta_distance)
    if os.path.isdir(base_folder):
        if not overwrite:
            raise Exception(
                "Folder already exists, if you want to overwrite it set the argument '-O' to 'true'"
            )
        else:
            shutil.rmtree(base_folder)
    os.mkdir(base_folder)
    for n in range(n_envs):
        base_env_folder = os.path.join(base_folder, f"env_{n+1:03d}")
        os.mkdir(base_env_folder)
        tunnel_network_params = TunnelNetworkParams.from_defaults()
        tunnel_network_params.min_distance_between_intersections = 30
        tunnel_network_params.collision_distance = 15
        tunnel_network = TunnelNetwork(params=tunnel_network_params)
        for _ in range(n_grown):
            GrownTunnelGenerationParams._random_distance_range = (100, 300)
            GrownTunnelGenerationParams._random_horizontal_tendency_range_deg = (
                -50,
                50,
            )
            GrownTunnelGenerationParams._random_horizontal_noise_range_deg = (-30, 30)
            GrownTunnelGenerationParams._random_min_segment_length_fraction_range = (
                0.05,
                0.05,
            )
            GrownTunnelGenerationParams._random_max_segment_length_fraction_range = (
                0.10,
                0.10,
            )
            result = False
            while not result:
                params = GrownTunnelGenerationParams.random()
                result = tunnel_network.add_random_grown_tunnel(
                    params=params, n_trials=100
                )
        for _ in range(n_connector):
            tunnel_network.add_random_connector_tunnel(n_trials=100)
        ptcl_gen_params = TunnelNetworkPtClGenParams.random()
        mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
        mesh_gen_params.fta_distance = fta_dist
        mesh_generator = TunnelNetworkMeshGenerator(
            tunnel_network,
            ptcl_gen_params=ptcl_gen_params,
            meshing_params=mesh_gen_params,
        )
        mesh_generator.compute_all()
        axis_points = gen_axis_points_file(mesh_generator)
        path_to_mesh = os.path.join(base_env_folder, "mesh.obj")
        mesh_generator.save_mesh(path_to_mesh)
        np.savetxt(os.path.join(base_env_folder, "axis.txt"), axis_points)
        np.savetxt(os.path.join(base_env_folder, "fta_dist.txt"), np.array((fta_dist,)))
        sdf = MODEL_SDF_TEXT.format(path_to_mesh, path_to_mesh)
        path_to_model_sdf = os.path.join(base_env_folder, "model.sdf")
        with open(path_to_model_sdf, "w") as f:
            f.write(sdf)


if __name__ == "__main__":
    main()
