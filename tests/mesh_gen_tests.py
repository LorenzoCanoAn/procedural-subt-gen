from subt_proc_gen.mesh_generation import (
    CylindricalPerlinNoiseMapper,
    CylindricalPerlinNoiseMapperParms,
    TunnelNewtorkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
)
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    Tunnel,
    GrownTunnelGenerationParams,
    ConnectorTunnelGenerationParams,
)
from subt_proc_gen.graph import Node
from subt_proc_gen.geometry import Point3D, Vector3D, get_two_perpendicular_vectors
from subt_proc_gen.display_functions import (
    plot_tunnel_ptcls,
    plot_graph,
    plot_intersection_ptcls,
    plot_splines,
    plot_ptcl,
    plot_mesh,
)
import numpy as np
from multiprocessing import Pool
from time import perf_counter_ns
from traceback import print_exc
import pyvista as pv
import logging as log

log.basicConfig(level=log.WARNING)


def timeit(function, **args):
    start = perf_counter_ns()
    result = function(**args)
    end = perf_counter_ns()
    elapsed = (end - start) * 1e-9
    print(f"{function.__name__} took {elapsed:.5f} secs")
    return result


def generate_coords(scale):
    coords = [] * scale**2
    for i in range(scale):
        for j in range(scale):
            coords.append(np.array((i, j)))
    return coords


def noise_with_pool(generator: CylindricalPerlinNoiseMapper, coords):
    return generator(coords)


def fill_image_with_noise(image, noise, coords):
    for n, c in zip(noise, coords):
        image[c[0], c[1]] = n
    return image


def test1():
    scale = 100
    generator = CylindricalPerlinNoiseMapper(
        scale, CylindricalPerlinNoiseMapperParms.from_defaults()
    )
    coords = generate_coords(scale=scale)
    noise = noise_with_pool(generator=generator, coords=coords)
    image = np.zeros((scale, scale))
    image = fill_image_with_noise(image=image, noise=noise, coords=coords)


def test2():
    tunnel_network = Tunnel
    tunnel_network = TunnelNetwork()
    first_node = Node((0, 0, 0))
    tunnel_network.add_node(first_node)
    first_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(0), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(first_tunnel)
    second_tunnel = Tunnel.grown(
        i_node=first_node,
        i_direction=Vector3D.from_inclination_yaw_length(
            inclination=0, yaw=np.deg2rad(90), length=30
        ),
        params=GrownTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(second_tunnel)
    third_tunnel = Tunnel.connector(
        i_node=first_tunnel[-2],
        f_node=second_tunnel[-2],
        params=ConnectorTunnelGenerationParams.from_defaults(),
    )
    tunnel_network.add_tunnel(third_tunnel)
    mesh_generator = TunnelNewtorkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=TunnelNetworkPtClGenParams.from_defaults(),
        meshing_params=TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()


def test3():
    tunnel_network = TunnelNetwork()
    tunnel_network.add_random_grown_tunnel()
    tunnel_network.add_random_grown_tunnel()
    tunnel_network.add_random_grown_tunnel()
    tunnel_network.add_random_grown_tunnel()
    tunnel_network.add_random_connector_tunnel(n_trials=100)
    mesh_generator = TunnelNewtorkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=TunnelNetworkPtClGenParams.from_defaults(),
        meshing_params=TunnelNetworkMeshGenParams.from_defaults(),
    )
    mesh_generator.compute_all()
    mesh_generator.save_mesh("mesh.obj")
    print(len(mesh_generator.mesh.points))


def main():
    tests = [test1, test2, test3]
    for test in tests:
        try:
            timeit(test)
        except:
            print(f"{test.__name__} failed")
            print_exc()


if __name__ == "__main__":
    main()
