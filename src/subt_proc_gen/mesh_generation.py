import numpy as np
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.geometry import (
    get_two_perpendicular_vectors,
    get_close_points_indices,
    get_close_points_to_point,
    get_uniform_points_in_sphere,
    Point3D,
    warp_angle_pi,
    distance_matrix,
)
from subt_proc_gen.perlin import (
    CylindricalPerlinNoiseMapper,
    SphericalPerlinNoiseMapper,
)
from subt_proc_gen.mesh_generation_params import (
    TunnelNetworkPtClGenStrategies,
    TunnelPtClGenParams,
    IntersectionPtClType,
    IntersectionPtClGenParams,
    TunnelNetworkPtClGenParams,
    MeshingApproaches,
    TunnelNetworkMeshGenParams,
)
from enum import Enum
import logging as log
import pyvista as pv
import open3d as o3d
import os
import math


class PtclVoxelizator:
    def __init__(self, ptcl, apss, voxel_size=5):
        self.voxel_size = voxel_size
        self.grid = dict()
        max_x = max(ptcl[:, 0])
        min_x = min(ptcl[:, 0])
        max_y = max(ptcl[:, 1])
        min_y = min(ptcl[:, 1])
        max_z = max(ptcl[:, 2])
        min_z = min(ptcl[:, 2])
        max_i = int(np.ceil(max_x / voxel_size)) + 3
        min_i = int(np.floor(min_x / voxel_size)) - 3
        max_j = int(np.ceil(max_y / voxel_size)) + 3
        min_j = int(np.floor(min_y / voxel_size)) - 3
        max_k = int(np.ceil(max_z / voxel_size)) + 3
        min_k = int(np.floor(min_z / voxel_size)) - 3
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                for k in range(min_k, max_k):
                    self.grid[(i, j, k)] = np.zeros([0, 6])
        ijks = np.floor(self.axis_points / voxel_size).astype(int)
        for ijk, ap, av in zip(ijks, self.axis_points, self.axis_vectors):
            i, j, k = ijk
            ap = np.reshape(ap, (-1, 3))
            av = np.reshape(av, (-1, 3))
            self.grid[(i, j, k)] = np.concatenate(
                [self.grid[(i, j, k)], np.concatenate((ap, av), axis=1)], axis=0
            )

    def get_relevant_points(self, xyz):
        _i, _j, _k = np.floor(xyz / self.voxel_size).astype(int)
        relevant_points = np.zeros((0, 6))
        for i in (_i - 1, _i, _i + 1):
            for j in (_j - 1, _j, _j + 1):
                for k in (_k - 1, _k, _k + 1):
                    relevant_points = np.concatenate(
                        [relevant_points, self.grid[(i, j, k)]], axis=0
                    )
        return relevant_points


class TunnelNewtorkMeshGenerator:
    def __init__(
        self,
        tunnel_network: TunnelNetwork,
        ptcl_gen_params: TunnelNetworkPtClGenParams,
        meshing_params: TunnelNetworkMeshGenParams,
    ):
        self._tunnel_network = tunnel_network
        self._ptcl_gen_params = ptcl_gen_params
        self._meshing_params = meshing_params
        self._params_of_intersections = dict()
        self._params_of_tunnels = dict()
        self._ptcl_of_tunnels = dict()
        self._normals_of_tunnels = dict()
        self._aps_avs_of_tunnels = dict()
        self._apss_avss_of_tunnels = dict()
        self._perlin_generator_of_tunnel = dict()
        self._ptcl_of_intersections = dict()
        self._normals_of_intersections = dict()
        self._aps_avs_of_intersections = dict()
        self._apss_avss_of_intersections = dict()
        self._radius_of_intersections_for_tunnels = dict()
        self._voxelization_of_ptcl = None

    @property
    def axis_points(self) -> np.ndarray:
        axis_points = np.zeros((0, 3))
        for t in self._aps_of_tunnels:
            axis_points = np.concatenate([axis_points, self._aps_of_tunnels[t]], axis=0)
        for intersection in self._aps_of_intersections:
            for elm in self._aps_of_intersections[intersection]:
                axis_points = np.concatenate(
                    [axis_points, self._aps_of_intersections[intersection][elm]]
                )
        return axis_points

    def params_of_intersection(self, intersection) -> IntersectionPtClGenParams:
        return self._params_of_intersections[intersection]

    def params_of_tunnel(self, tunnel) -> TunnelPtClGenParams:
        return self._params_of_tunnels[tunnel]

    def _compute_intersection_ptcl(
        self,
        ptcl_gen_params: IntersectionPtClGenParams,
    ):
        raise NotImplementedError()

    def _set_params_of_each_tunnel_ptcl_gen(self, ptcl_gen_params=None):
        if ptcl_gen_params is None:
            ptcl_gen_params = self._ptcl_gen_params
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.default:
            for tunnel in self._tunnel_network.tunnels:
                if tunnel in ptcl_gen_params.pre_set_tunnel_params:
                    self._params_of_tunnels[
                        tunnel
                    ] = ptcl_gen_params.pre_set_tunnel_params[tunnel]
                else:
                    self._params_of_tunnels[
                        tunnel
                    ] = TunnelPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.random:
            for tunnel in self._tunnel_network.tunnels:
                if tunnel in ptcl_gen_params.pre_set_tunnel_params:
                    self._params_of_tunnels[
                        tunnel
                    ] = ptcl_gen_params.pre_set_tunnel_params[tunnel]
                else:
                    self._params_of_tunnels[tunnel] = TunnelPtClGenParams.random()
        if not ptcl_gen_params.general_fta_distance is None:
            for tunnel in self._tunnel_network.tunnels:
                self._params_of_tunnels[tunnel].flatten_floor = True
                self._params_of_tunnels[
                    tunnel
                ].fta_distance = ptcl_gen_params.general_fta_distance

    def _set_params_of_each_intersection_ptcl_gen(
        self, ptcl_gen_params: TunnelNetworkPtClGenParams = None
    ):
        if ptcl_gen_params is None:
            ptcl_gen_params = self._ptcl_gen_params
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.default:
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.random:
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.random()
        if not ptcl_gen_params.general_fta_distance is None:
            for intersection in self._tunnel_network.intersections:
                self._params_of_intersections[intersection].flatten_floor = True
                self._params_of_intersections[
                    intersection
                ].fta_distance = ptcl_gen_params.general_fta_distance

    def _set_perlin_mappers_of_tunnels(self):
        for tunnel in self._tunnel_network.tunnels:
            self._perlin_generator_of_tunnel[tunnel] = CylindricalPerlinNoiseMapper(
                sampling_scale=tunnel.spline.metric_length,
                params=self.params_of_tunnel(tunnel).perlin_params,
            )

    def _compute_tunnel_ptcls(self):
        for tunnel in self._tunnel_network.tunnels:
            (
                self._ptcl_of_tunnels[tunnel],
                self._normals_of_tunnels[tunnel],
                self._aps_of_tunnels[tunnel],
                self._avs_of_tunnels[tunnel],
                self._apss_of_tunnels[tunnel],
            ) = ptcl_from_tunnel(
                tunnel=tunnel,
                perlin_mapper=self._perlin_generator_of_tunnel[tunnel],
                dist_between_circles=self.params_of_tunnel(tunnel).dist_between_circles,
                n_points_per_circle=self.params_of_tunnel(tunnel).n_points_per_circle,
                radius=self.params_of_tunnel(tunnel).radius,
                noise_magnitude=self.params_of_tunnel(tunnel).noise_relative_magnitude,
                perlin_weight_angle=self.params_of_tunnel(
                    tunnel
                ).perlin_weight_by_angle,
            )

    def _compute_radius_of_intersections_for_all_tunnels(self):
        for intersection in self._tunnel_network.intersections:
            for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                intersection_radius_for_tunnel_at_intersection = 0
                for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                    # Tunnel i is the one being evaluated here
                    if tunnel_i is tunnel_j:
                        continue
                    intersection_radius_for_tunnel_at_intersection = max(
                        intersection_radius_for_tunnel_at_intersection,
                        self.get_safe_radius_from_intersection_of_two_tunnels(
                            tunnel_i,
                            tunnel_j,
                            intersection,
                            starting_radius=self.params_of_intersection(
                                intersection
                            ).radius,
                        ),
                    )
                self._radius_of_intersections_for_tunnels[
                    (intersection, tunnel_i)
                ] = intersection_radius_for_tunnel_at_intersection

    def _separate_intersection_ptcl_from_tunnel_ptcls(self):
        for intersection in self._tunnel_network.intersections:
            tunnels_of_intersection = self._tunnel_network._tunnels_of_node[
                intersection
            ]
            pts_of_intersection = dict()
            ns_of_intersection = dict()
            aps_of_intersection = dict()
            avs_of_intersection = dict()
            apss_of_intersection = dict()
            for tunnel in tunnels_of_intersection:
                tunnel_pts = self._ptcl_of_tunnels[tunnel]
                tunnel_ns = self._normals_of_tunnels[tunnel]
                tunnel_aps = self._aps_of_tunnels[tunnel]
                tunnel_avs = self._avs_of_tunnels[tunnel]
                tunnel_apss = self._apss_of_tunnels[tunnel]
                radius = self._radius_of_intersections_for_tunnels[
                    (intersection, tunnel)
                ]
                idxs_of_t_ptcl_to_int_ptcl = get_close_points_indices(
                    intersection.xyz, tunnel_pts, radius
                )
                idxs_of_aps_from_t_to_int = get_close_points_indices(
                    intersection.xyz,
                    tunnel_aps,
                    radius,
                )
                pts_of_intersection[tunnel] = np.reshape(
                    tunnel_pts[idxs_of_t_ptcl_to_int_ptcl, :], (-1, 3)
                )
                ns_of_intersection[tunnel] = np.reshape(
                    tunnel_ns[idxs_of_t_ptcl_to_int_ptcl, :], (-1, 3)
                )
                apss_of_intersection[tunnel] = np.reshape(
                    tunnel_apss[idxs_of_t_ptcl_to_int_ptcl, :], (-1, 3)
                )
                aps_of_intersection[tunnel] = np.reshape(
                    tunnel_aps[idxs_of_aps_from_t_to_int, :],
                    (-1, 3),
                )
                avs_of_intersection[tunnel] = np.reshape(
                    tunnel_avs[idxs_of_aps_from_t_to_int, :],
                    (-1, 3),
                )
                self._ptcl_of_tunnels[tunnel] = np.delete(
                    self._ptcl_of_tunnels[tunnel], idxs_of_t_ptcl_to_int_ptcl, axis=0
                )
                self._normals_of_tunnels[tunnel] = np.delete(
                    self._normals_of_tunnels[tunnel], idxs_of_t_ptcl_to_int_ptcl, axis=0
                )
                self._aps_of_tunnels[tunnel] = np.delete(
                    self._aps_of_tunnels[tunnel], idxs_of_aps_from_t_to_int, axis=0
                )
                self._avs_of_tunnels[tunnel] = np.delete(
                    self._avs_of_tunnels[tunnel], idxs_of_aps_from_t_to_int, axis=0
                )
                self._apss_of_tunnels[tunnel] = np.delete(
                    self._apss_of_tunnels[tunnel], idxs_of_t_ptcl_to_int_ptcl, axis=0
                )
            self._ptcl_of_intersections[intersection] = pts_of_intersection
            self._normals_of_intersections[intersection] = ns_of_intersection
            self._aps_of_intersections[intersection] = aps_of_intersection
            self._avs_of_intersections[intersection] = avs_of_intersection
            self._apss_of_intersections[intersection] = apss_of_intersection

    def ptcl_of_intersection(self, intersection: Node):
        return np.concatenate(
            [
                self._ptcl_of_intersections[intersection][element]
                for element in self._ptcl_of_intersections[intersection]
            ],
            axis=0,
        )

    def normals_of_intersection(self, intersection: Node):
        return np.concatenate(
            [
                self._normals_of_intersections[intersection][element]
                for element in self._normals_of_intersections[intersection]
            ],
            axis=0,
        )

    @property
    def ptcl(self):
        global_ptcl = np.zeros((0, 3))
        for intersection in self._tunnel_network.intersections:
            for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                ptcl = self._ptcl_of_intersections[intersection][tunnel]
                global_ptcl = np.vstack((global_ptcl, ptcl))
        for tunnel in self._tunnel_network.tunnels:
            ptcl = self._ptcl_of_tunnels[tunnel]
            global_ptcl = np.vstack((global_ptcl, ptcl))
        return global_ptcl

    @property
    def ptcl_apss(self):
        ptcl_apss = np.zeros((0, 6))
        for intersection in self._tunnel_network.intersections:
            for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                ptcl = self._ptcl_of_intersections[intersection][tunnel]
                apss = self._apss_of_intersections[intersection][tunnel]
                ptcl_apss = np.vstack((ptcl_apss, np.hstack((ptcl, apss))))
        for tunnel in self._tunnel_network.tunnels:
            ptcl = self._ptcl_of_tunnels[tunnel]
            apss = self._apss_of_tunnels[tunnel]
            ptcl_apss = np.vstack((ptcl_apss, np.hstack((ptcl, apss))))
        return ptcl_apss

    @property
    def n_of_intersection_normals(self):
        n_normals = 0
        for intersection in self._tunnel_network.intersections:
            n_normals += self.normals_of_intersection(intersection).shape[0]
        return n_normals

    @property
    def n_of_tunel_normals(self):
        n_normals = 0
        for intersection in self._tunnel_network.tunnels:
            n_normals += self.normals_of_tunnel(intersection).shape[0]
        return n_normals

    @property
    def n_of_intersection_points(self):
        n_points = 0
        for intersection in self._tunnel_network.intersections:
            n_points += self.ptcl_of_intersection(intersection).shape[0]
        return n_points

    @property
    def n_of_tunel_points(self):
        n_points = 0
        for tunnel in self._tunnel_network.tunnels:
            n_points += self.ptcl_of_tunnel(tunnel).shape[0]
        return n_points

    @property
    def n_of_normals(self):
        return self.n_of_intersection_normals + self.n_of_tunel_normals

    @property
    def n_of_points(self):
        return self.n_of_tunel_points + self.n_of_intersection_points

    @property
    def complete_normals(self):
        complete_normals = np.zeros((self.n_of_normals, 3))
        n_normals = 0
        for intersection in self._tunnel_network.intersections:
            normals_of_intersection = self.normals_of_intersection(intersection)
            complete_normals[
                n_normals : n_normals + normals_of_intersection.shape[0]
            ] = normals_of_intersection
            n_normals += normals_of_intersection.shape[0]
        for tunnel in self._tunnel_network.tunnels:
            normals_of_tunnel = self.normals_of_tunnel(tunnel)
            complete_normals[
                n_normals : n_normals + normals_of_tunnel.shape[0]
            ] = normals_of_tunnel
            n_normals += normals_of_tunnel.shape[0]
        return complete_normals

    def ptcl_of_tunnel(self, tunnel: Tunnel) -> np.ndarray:
        return self._ptcl_of_tunnels[tunnel]

    def normals_of_tunnel(self, tunnel: Tunnel) -> np.ndarray:
        return self._normals_of_tunnels[tunnel]

    def clean_tunnels_in_intersection(self, intersection: Node):
        # Get the ids of the points to delete
        idxs_of_tunnel_pts_to_delete = dict()
        for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
            ids_to_delete_from_i = set()
            for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                if tunnel_i is tunnel_j:
                    continue
                ids_of_i_in_j = np.reshape(
                    points_inside_of_tunnel_section(
                        self._aps_of_intersections[intersection][tunnel_j],
                        self._ptcl_of_intersections[intersection][tunnel_j],
                        self._ptcl_of_intersections[intersection][tunnel_i],
                        self._avs_of_intersections[intersection][tunnel_j],
                    ),
                    -1,
                )
                for idx in ids_of_i_in_j:
                    ids_to_delete_from_i.add(idx)
            idxs_of_tunnel_pts_to_delete[tunnel_i] = np.array(
                list(ids_to_delete_from_i)
            )
        for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
            if len(idxs_of_tunnel_pts_to_delete[tunnel]) == 0:
                continue
            self._ptcl_of_intersections[intersection][tunnel] = np.reshape(
                np.delete(
                    self._ptcl_of_intersections[intersection][tunnel],
                    idxs_of_tunnel_pts_to_delete[tunnel],
                    axis=0,
                ),
                (-1, 3),
            )
            self._normals_of_intersections[intersection][tunnel] = np.reshape(
                np.delete(
                    self._normals_of_intersections[intersection][tunnel],
                    idxs_of_tunnel_pts_to_delete[tunnel],
                    axis=0,
                ),
                (-1, 3),
            )
            self._apss_of_intersections[intersection][tunnel] = np.reshape(
                np.delete(
                    self._apss_of_intersections[intersection][tunnel],
                    idxs_of_tunnel_pts_to_delete[tunnel],
                    axis=0,
                ),
                (-1, 3),
            )

    def _compute_all_intersections_ptcl(self):
        for intersection in self._tunnel_network.intersections:
            params = self.params_of_intersection(intersection)
            if params.ptcl_type == IntersectionPtClType.no_cavity:
                print("cleaning tunnels")
                self.clean_tunnels_in_intersection(intersection)
            elif params.ptcl_type == IntersectionPtClType.spherical_cavity:
                self.clean_tunnels_in_intersection(intersection)
                center_point = intersection.xyz
                radius = params.radius
                points_per_sm = params.points_per_sm
                # Generate the points of the intersection
                sphere_points, sphere_normals = generate_noisy_sphere(
                    center_point,
                    radius,
                    points_per_sm,
                    SphericalPerlinNoiseMapper(params.perlin_params),
                    params.noise_multiplier,
                )
                ids_to_delete_in_cavity = set()
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                    axis_of_tunnel = self._aps_of_intersections[intersection][tunnel]
                    points_of_tunnel = self._ptcl_of_intersections[intersection][tunnel]
                    ids_to_delete_in_tunnel = ids_points_inside_ptcl_sphere(
                        sphere_points, center_point, points_of_tunnel
                    )
                    ids_to_delete_in_cavity_ = points_inside_of_tunnel_section(
                        axis_of_tunnel, points_of_tunnel, sphere_points
                    )
                    self._ptcl_of_intersections[intersection][tunnel] = np.delete(
                        self._ptcl_of_intersections[intersection][tunnel],
                        ids_to_delete_in_tunnel,
                        axis=0,
                    )
                    self._normals_of_intersections[intersection][tunnel] = np.delete(
                        self._normals_of_intersections[intersection][tunnel],
                        ids_to_delete_in_tunnel,
                        axis=0,
                    )
                    for id_ in np.reshape(ids_to_delete_in_cavity_, -1):
                        ids_to_delete_in_cavity.add(id_)
                sphere_points = np.delete(
                    sphere_points, np.array(list(ids_to_delete_in_cavity)), axis=0
                )
                sphere_normals = np.delete(
                    sphere_normals, np.array(list(ids_to_delete_in_cavity)), axis=0
                )
                self._ptcl_of_intersections[intersection]["cavity"] = sphere_points
                self._normals_of_intersections[intersection]["cavity"] = sphere_normals

    def get_safe_radius_from_intersection_of_two_tunnels(
        self,
        tunnel_i: Tunnel,
        tunnel_j: Tunnel,
        intersection: Node,
        increment=2,
        starting_radius=20,
    ):
        assert intersection in tunnel_i.nodes
        assert intersection in tunnel_j.nodes
        tpi = self.ptcl_of_tunnel(tunnel_i)
        tpj = self.ptcl_of_tunnel(tunnel_j)
        _, apj, avj = tunnel_j.spline.discretize(
            self.params_of_tunnel(tunnel_j).dist_between_circles
        )
        cut_off_radius = starting_radius
        while True:
            aidx = get_close_points_indices(intersection.xyz, apj, cut_off_radius)
            apj_to_use = np.reshape(apj[aidx, :], (-1, 3))
            tpj_to_use = get_close_points_to_point(
                intersection.xyz, tpj, cut_off_radius
            )
            tpi_to_use = get_close_points_to_point(
                intersection.xyz, tpi, cut_off_radius
            )
            id_of_p_of_i_inside_j = points_inside_of_tunnel_section(
                apj_to_use, tpj_to_use, tpi_to_use
            )
            p_of_i_inside_j = np.reshape(tpi_to_use[id_of_p_of_i_inside_j, :], (-1, 3))
            dist_of_i_points_inside_j_to_inter = np.linalg.norm(
                intersection.xyz - p_of_i_inside_j, axis=1
            )
            if len(dist_of_i_points_inside_j_to_inter) == 0:
                radius = cut_off_radius
                break
            radius = np.max(dist_of_i_points_inside_j_to_inter)
            if radius < cut_off_radius - increment:
                break
            cut_off_radius += increment
        return max(radius + increment, starting_radius)

    def compute_all(self):
        log.info("Setting parameters of tunnels")
        self._set_params_of_each_tunnel_ptcl_gen()
        log.info("Setting perlin mappers for tunnels")
        self._set_perlin_mappers_of_tunnels()
        log.info("Computing pointclouds of tunnels")
        self._compute_tunnel_ptcls()
        log.info("Setting paramters for intersections")
        self._set_params_of_each_intersection_ptcl_gen()
        log.info("Calcludating radius of intersections")
        self._compute_radius_of_intersections_for_all_tunnels()
        log.info("Separating the pointclouds of the tunnels for the intersections")
        self._separate_intersection_ptcl_from_tunnel_ptcls()
        log.info("Computing pointclouds of intersections")
        self._compute_all_intersections_ptcl()
        log.info("Computing mesh")
        self._compute_mesh()
        log.info("Voxelizing ptcl")
        self._voxelize_ptcl()
        log.info("Flattening floors")
        self._flatten_floors()

    def _compute_mesh(self):
        points = self.ptcl
        normals = self.complete_normals
        if self._meshing_params.meshing_approach == MeshingApproaches.poisson:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self._meshing_params.poisson_depth
            )
            log.info("Simplifying mesh")
            simplified_mesh = o3d.geometry.simplify_vertex_clustering(
                o3d_mesh, self._meshing_params.simplification_voxel_size
            )
            o3d.io.write_triangle_mesh("mesh.obj", simplified_mesh)
            self.mesh = pv.read("mesh.obj")
            os.remove("mesh.obj")
        else:
            raise NotImplementedError(
                f"The method {self._meshing_params.meshing_approach} is not implemented"
            )

    def _voxelize_ptcl(self):
        self._voxelization_of_ptcl = PtclVoxelizator(
            self.ptcl,
        )

    def save_mesh(self, path):
        pv.save_meshio(path, self.mesh)

    def perlin_generator_of_tunnel(
        self, tunnel: Tunnel
    ) -> CylindricalPerlinNoiseMapper:
        return self._perlin_generator_of_tunnel[tunnel]


#########################################################################################################################
# Functions
#########################################################################################################################


def ptcl_from_tunnel(
    tunnel: Tunnel,
    perlin_mapper,
    dist_between_circles,
    n_points_per_circle,
    radius,
    noise_magnitude,
    perlin_weight_angle,
    d_min=None,
    d_max=None,
):
    ads, aps, avs = tunnel.spline.discretize(dist_between_circles, d_min, d_max)
    n_circles = aps.shape[0]
    angs = np.linspace(0, 2 * np.pi, n_points_per_circle)
    pws = perlin_weight_from_angle(angs, perlin_weight_angle)
    angss = np.concatenate([angs for _ in range(n_circles)])
    pwss = np.concatenate([pws for _ in range(n_circles)])
    angss = np.reshape(angss, [-1, 1])
    pwss = np.reshape(pwss, [-1, 1])
    apss = np.concatenate(
        [np.full((n_points_per_circle, 3), aps[i, :]) for i in range(n_circles)],
        axis=0,
    )
    us = np.zeros(avs.shape)
    vs = np.zeros(avs.shape)
    for i, av in enumerate(avs):
        u, v = get_two_perpendicular_vectors(av)
        us[i, :] = np.reshape(u.cartesian_unitary, -1)
        vs[i, :] = np.reshape(v.cartesian_unitary, -1)
    uss = np.concatenate(
        [np.full((n_points_per_circle, 3), us[i, :]) for i in range(n_circles)],
        axis=0,
    )
    vss = np.concatenate(
        [np.full((n_points_per_circle, 3), vs[i, :]) for i in range(n_circles)],
        axis=0,
    )
    normals = uss * np.sin(angss) + vss * np.cos(angss)
    dss = np.concatenate(
        [np.full((n_points_per_circle, 1), ads[i, :]) for i in range(n_circles)],
        axis=0,
    )
    archdss = angss * radius
    cylindrical_coords = np.concatenate([dss, archdss], axis=1)
    noise_to_add = perlin_mapper(cylindrical_coords)
    noise_to_add /= np.max(np.abs(noise_to_add))
    points = (
        apss
        + normals * radius
        + normals * radius * noise_magnitude * noise_to_add * pwss
    )
    return points, normals, aps, avs, apss


def generate_noisy_sphere(
    center_point,
    radius,
    points_per_sm,
    perlin_mapper: SphericalPerlinNoiseMapper,
    noise_multiplier,
):
    area_of_sphere = 4 * np.pi * radius**2
    n_points = int(math.ceil(points_per_sm * area_of_sphere))
    points_before_noise = get_uniform_points_in_sphere(n_points)
    noise_of_points = np.reshape(perlin_mapper(points_before_noise), (-1, 1))
    noise_of_points /= np.max(np.abs(noise_of_points))
    noise_of_points -= 1
    noise_of_points /= 2
    points_with_noise = (
        points_before_noise + noise_multiplier * noise_of_points * points_before_noise
    )
    points_with_noise_and_radius = points_with_noise * radius
    normals = points_with_noise_and_radius / np.reshape(
        np.linalg.norm(points_with_noise_and_radius, axis=1), (-1, 1)
    )
    points = points_with_noise_and_radius + center_point
    return points, normals


def points_inside_of_tunnel_section(
    axis_points: np.ndarray,
    tunnel_points: np.ndarray,
    points: np.ndarray,
    axis_vectors: np.ndarray = None,
    max_projection_over_axis=0.5,
):
    try:
        assert axis_points.shape[0] > 0
        assert axis_points.shape[1] == 3
        assert tunnel_points.shape[0] > 0
        assert tunnel_points.shape[1] == 3
        assert points.shape[0] > 0
        assert points.shape[1] == 3
    except:
        return np.array([], dtype=int)
    dist_of_ps_to_aps = distance_matrix(points, axis_points)
    closest_ap_to_p_idx = np.argmin(dist_of_ps_to_aps, axis=1)
    if not axis_vectors is None:
        closest_ap_to_p_vector = points - axis_points[closest_ap_to_p_idx, :]
        ap_vector_of_p = axis_vectors[closest_ap_to_p_idx, :]
        projection = np.array(
            [
                np.dot(ap_vector_of_p[i, :], closest_ap_to_p_vector[i, :])
                for i in range(ap_vector_of_p.shape[0])
            ]
        )
        outside_because_of_angle = np.abs(projection) > max_projection_over_axis
    else:
        outside_because_of_angle = np.zeros((points.shape[0]), dtype=np.bool8)
    dist_of_ps_to_tps = distance_matrix(points, tunnel_points)
    dist_of_tps_to_aps = distance_matrix(tunnel_points, axis_points)
    dist_to_ap_of_p = np.min(dist_of_ps_to_aps, axis=1)
    closest_tp_to_p = np.argmin(dist_of_ps_to_tps, axis=1)
    inside_for_radius = (
        dist_to_ap_of_p < dist_of_tps_to_aps[closest_tp_to_p, closest_ap_to_p_idx]
    )
    inside = np.logical_and(inside_for_radius, np.logical_not(outside_because_of_angle))
    return np.where(inside)


def ids_points_inside_ptcl_sphere(sphere_points, center_point, points):
    dists_of_ps_to_sph_pts = distance_matrix(points, sphere_points)
    dists_of_sph_pts_to_center = np.reshape(
        np.linalg.norm(sphere_points - center_point, axis=1), (-1, 1)
    )
    dists_of_ps_to_center = np.reshape(
        np.linalg.norm(points - center_point, axis=1), (-1, 1)
    )
    id_closest_sph_pt_to_pt = np.argmin(dists_of_ps_to_sph_pts, axis=1)
    return np.where(
        dists_of_ps_to_center < dists_of_sph_pts_to_center[id_closest_sph_pt_to_pt]
    )


def perlin_weight_from_angle(angles_rad, perlin_weight_angle_rad):
    """Given an set of angles in radians, and a cuttoff angle, this function returns the weight
    that the perlin noise should have in a given angle, so that there is no discontinuity in the resulting image"""
    perlin_weights = np.zeros(angles_rad.shape)
    for i, angle in enumerate(angles_rad):
        warped_angle = warp_angle_pi(angle)
        if abs(warped_angle) < perlin_weight_angle_rad:
            perlin_weights[i] = (abs(warped_angle) / perlin_weight_angle_rad) ** 2
        else:
            perlin_weights[i] = 1
    return perlin_weights
