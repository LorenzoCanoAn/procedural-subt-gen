import os
import math
import numpy as np
from perlin_noise import PerlinNoise
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Intersection
import open3d as o3d
from time import time_ns as ns
import random


class TunnelPtClGenParams:
    # Default params
    _default_radius = 4
    _default_noise_freq = 0.1
    _default_noise_magn = 1
    _default_flatten_floor = True
    _default_fta_distance = 2
    # Random params
    _random_radius_interval = (1, 6)
    _random_noise_freq_interval = (0.3, 0)
    _random_noise_relative_magn_interval = (0.1, 0.7)
    _random_flatten_floor_probability = 0.5
    _random_fta_relative_distance_interval = [0, 1.5]

    @classmethod
    def from_defaults(cls):
        return TunnelPtClGenParams(
            radius=cls._default_radius,
            noise_freq=cls._default_noise_freq,
            noise_magn=cls._default_noise_magn,
            flatten_floor=cls._default_flatten_floor,
            fta_distance=cls._default_fta_distance,
        )

    @classmethod
    def random(cls):
        radius = np.random.uniform(
            cls._random_radius_interval[0],
            cls._random_radius_interval[1],
        )
        noise_freq = np.random.uniform(
            cls._random_noise_freq_interval[0],
            cls._random_noise_freq_interval[1],
        )
        noise_magn = radius * np.random.uniform(
            cls._random_noise_relative_magn_interval[0],
            cls._random_noise_relative_magn_interval[1],
        )
        flatten_floor = cls._random_flatten_floor_probability > np.random.random()
        fta_distance = radius * np.random.uniform(
            cls._random_noise_relative_magn_interval[0],
            cls._random_noise_relative_magn_interval[1],
        )
        return TunnelPtClGenParams(
            radius=radius,
            noise_freq=noise_freq,
            noise_magn=noise_magn,
            flatten_floor=flatten_floor,
            fta_distance=fta_distance,
        )

    def __init__(
        self,
        radius=None,
        noise_freq=None,
        noise_magn=None,
        flatten_floor=None,
        fta_distance=None,
    ):
        assert not radius is None
        assert not noise_freq is None
        assert not noise_magn is None
        assert not flatten_floor is None
        assert not fta_distance is None
        self.radius = radius
        self.noise_freq = noise_freq
        self.flatter_floor = flatten_floor
        self.fta_distance = fta_distance


class IntersectionPtClGenParams:
    """Params that control how the pointcloud of an intersection is generated"""


class TunnelNetworkPtClGenParams:
    """Params that control the the overall generation of the pointcloud of the
    complete tunnel network"""


class TunnelNetworkMeshGenParams:
    """Params that control how the mesh is generated from the ptcl"""


class TunnelNewtorkMeshGenerator:
    def __init__(
        self,
        tunnel_network: TunnelNetwork,
        meshing_params: TunnelNetworkMeshGenParams,
    ):
        self._tunnel_network = tunnel_network
        self._ptcl_of_tunnel = dict()
        self._ptcl_of_intersections = dict()

    def _compute_tunnel_ptcl(
        self,
        tunnel: Tunnel,
        ptcl_gen_params: TunnelPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_intersection_ptcl(
        self,
        intersection: Intersection,
        ptcl_gen_params: IntersectionPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_all_tunnels_ptcl(
        self,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
    ):
        raise NotImplementedError()

    def _compute_all_intersections_ptcl(self):
        raise NotImplementedError()

    def compute_all(
        self,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
        mesh_gen_params: TunnelNetworkMeshGenParams,
    ):
        self.compute_all_tunnel_ptcl(ptcl_gen_params)
        self.compute_all_intersection_ptcl(ptcl_gen_params)
        self.compute_mesh(mesh_gen_params)


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(tunnel: Tunnel, params: TunnelPtClGenParams):
    TODO
