import numpy as np
from subt_proc_gen.graph import Node
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.geometry import (
    get_two_perpendicular_vectors,
    get_close_points_indices,
    get_close_points_to_point,
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
        self._ptcl_of_tunnels = dict()
        self._normals_of_tunnels = dict()
        self._params_of_tunnels = dict()
        self._perlin_generator_of_tunnel = dict()
        self._ptcl_of_intersections = dict()
        self._normals_of_intersections = dict()
        self._aps_of_intersections = dict()
        self._avs_of_intersections = dict()
        self._params_of_intersections = dict()
        self._radius_of_intersections_for_tunnels = dict()

    def params_of_intersection(self, intersection) -> IntersectionPtClGenParams:
        return self._params_of_intersections[intersection]

    def params_of_tunnel(self, tunnel) -> TunnelPtClGenParams:
        return self._params_of_tunnels[tunnel]

    def _compute_intersection_ptcl(
        self,
        ptcl_gen_params: IntersectionPtClGenParams,
    ):
        raise NotImplementedError()

    def _set_params_of_each_tunnel_ptcl_gen(
        self, ptcl_gen_params: TunnelNetworkPtClGenParams
    ):
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

    def _set_params_of_each_intersection_ptcl_gen(
        self, ptcl_gen_params: TunnelNetworkPtClGenParams
    ):
        if ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.default:
            for intersection in self._tunnel_network.intersections:
                self._params_of_intersections[
                    intersection
                ] = IntersectionPtClGenParams.from_defaults()
        elif ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.random:
            for intersection in self._tunnel_network.intersections:
                self._params_of_intersections[
                    intersection
                ] = IntersectionPtClGenParams.random()
        elif (
            ptcl_gen_params.strategy
            == TunnelNetworkPtClGenStrategies.pre_set_or_default
        ):
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.from_defaults()
        elif (
            ptcl_gen_params.strategy == TunnelNetworkPtClGenStrategies.pre_set_or_random
        ):
            for intersection in self._tunnel_network.intersections:
                if intersection in ptcl_gen_params.pre_set_intersection_params:
                    self._params_of_intersections[
                        intersection
                    ] = ptcl_gen_params.pre_set_intersection_params[intersection]
                else:
                    self._params_of_intersections[
                        intersection
                    ] = IntersectionPtClGenParams.random()

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
                flatten_floors=self.params_of_tunnel(tunnel).flatter_floor,
                fta_distance=self.params_of_tunnel(tunnel).fta_distance,
            )

    def _compute_radius_of_intersections_for_all_tunnels(self):
        for intersection in self._tunnel_network.intersections:
            for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                    # Tunnel i is the one being evaluated here
                    if tunnel_i is tunnel_j:
                        continue
                    intersection_radius_for_tunnel_at_intersection = (
                        self.get_safe_radius_from_intersection_of_two_tunnels(
                            tunnel_i,
                            tunnel_j,
                            intersection,
                            starting_radius=self.params_of_intersection(
                                intersection
                            ).radius,
                        )
                    )
                self._radius_of_intersections_for_tunnels[
                    (intersection, tunnel_i)
                ] = intersection_radius_for_tunnel_at_intersection

    def _separate_intersection_ptcl_from_tunnel_ptcls(self):
        for intersection in self._tunnel_network.intersections:
            tunnels_of_intersection = self._tunnel_network._tunnels_of_node[
                intersection
            ]
            ptcl_of_intersection = dict()
            normals_of_intersection = dict()
            axis_of_intersection = dict()
            axisv_of_intersection = dict()
            for tunnel in tunnels_of_intersection:
                tunnel_ptcl = self._ptcl_of_tunnels[tunnel]
                tunnel_normals = self._normals_of_tunnels[tunnel]
                radius = max(
                    self.params_of_intersection(intersection).radius,
                    self._radius_of_intersections_for_tunnels[(intersection, tunnel)],
                )
                indices_of_points_of_intersection = get_close_points_indices(
                    intersection.xyz, tunnel_ptcl, radius
                )
                _, tunnel_axis_points, tunnel_axis_vectors = tunnel.spline.discretize(
                    self.params_of_tunnel(tunnel).dist_between_circles
                )
                indices_of_axis_points_of_intersection = get_close_points_indices(
                    intersection.xyz,
                    tunnel_axis_points,
                    radius,
                )
                ptcl_of_intersection[tunnel] = np.reshape(
                    tunnel_ptcl[indices_of_points_of_intersection, :], (-1, 3)
                )
                normals_of_intersection[tunnel] = np.reshape(
                    tunnel_normals[indices_of_points_of_intersection, :], (-1, 3)
                )
                axis_of_intersection[tunnel] = np.reshape(
                    tunnel_axis_points[indices_of_axis_points_of_intersection, :],
                    (-1, 3),
                )
                axisv_of_intersection[tunnel] = np.reshape(
                    tunnel_axis_vectors[indices_of_axis_points_of_intersection, :],
                    (-1, 3),
                )
                self._ptcl_of_tunnels[tunnel] = np.delete(
                    tunnel_ptcl, indices_of_points_of_intersection, axis=0
                )
                self._normals_of_tunnels[tunnel] = np.delete(
                    tunnel_normals, indices_of_points_of_intersection, axis=0
                )
            self._ptcl_of_intersections[intersection] = ptcl_of_intersection
            self._normals_of_intersections[intersection] = normals_of_intersection
            self._aps_of_intersections[intersection] = axis_of_intersection
            self._avs_of_intersections[intersection] = axisv_of_intersection

    def ptcl_of_intersection(self, intersection: Node):
        return np.concatenate(
            [
                self._ptcl_of_intersections[intersection][tunnel]
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]
            ],
            axis=0,
        )

    def normals_of_intersection(self, intersection: Node):
        return np.concatenate(
            [
                self._normals_of_intersections[intersection][tunnel]
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]
            ],
            axis=0,
        )

    @property
    def complete_pointcloud(self):
        complete_ptcl = np.zeros((self.n_of_points, 3))
        n_points = 0
        for intersection in self._tunnel_network.intersections:
            ptcl_of_intersection = self.ptcl_of_intersection(intersection)
            complete_ptcl[
                n_points : n_points + ptcl_of_intersection.shape[0]
            ] = ptcl_of_intersection
            n_points += ptcl_of_intersection.shape[0]
        for tunnel in self._tunnel_network.tunnels:
            ptcl_of_tunnel = self.ptcl_of_tunnel(tunnel)
            complete_ptcl[
                n_points : n_points + ptcl_of_tunnel.shape[0]
            ] = ptcl_of_tunnel
            n_points += ptcl_of_tunnel.shape[0]
        return complete_ptcl

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

    def _compute_all_intersections_ptcl(self):
        for intersection in self._tunnel_network.intersections:
            params = self.params_of_intersection(intersection)
            ids_to_delete = dict()
            if params.ptcl_type == IntersectionPtClType.no_cavity:
                # Get the ids of the points to delete
                for tunnel_i in self._tunnel_network._tunnels_of_node[intersection]:
                    for tunnel_j in self._tunnel_network._tunnels_of_node[intersection]:
                        if tunnel_i is tunnel_j:
                            continue
                        ids_to_delete[tunnel_j] = points_inside_of_tunnel_section(
                            self._aps_of_intersections[intersection][tunnel_i],
                            self._ptcl_of_intersections[intersection][tunnel_i],
                            self._ptcl_of_intersections[intersection][tunnel_j],
                            self._avs_of_intersections[intersection][tunnel_i],
                        )
                # Delete the ids
                for tunnel in self._tunnel_network._tunnels_of_node[intersection]:
                    self._ptcl_of_intersections[intersection][tunnel] = np.reshape(
                        np.delete(
                            self._ptcl_of_intersections[intersection][tunnel],
                            ids_to_delete[tunnel],
                            axis=0,
                        ),
                        (-1, 3),
                    )
                    self._normals_of_intersections[intersection][tunnel] = np.reshape(
                        np.delete(
                            self._normals_of_intersections[intersection][tunnel],
                            ids_to_delete[tunnel],
                            axis=0,
                        ),
                        (-1, 3),
                    )
            elif params.ptcl_type == IntersectionPtClType.spherical_cavity:
                center_point = intersection.xyz
                radius = params.radius
                points_per_sm = params.point_density
                sphere_points = generate_noisy_sphere(
                    center_point,
                    radius,
                    points_per_sm,
                    SphericalPerlinNoiseMapper(params.perlin_params),
                )

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
            radius = np.max(dist_of_i_points_inside_j_to_inter)
            if radius < cut_off_radius - increment:
                break
            cut_off_radius += increment
        return radius + increment

    def compute_all(self):
        log.info("Setting parameters of tunnels")
        self._set_params_of_each_tunnel_ptcl_gen(self._ptcl_gen_params)
        log.info("Setting perlin mappers for tunnels")
        self._set_perlin_mappers_of_tunnels()
        log.info("Computing pointclouds of tunnels")
        self._compute_tunnel_ptcls()
        log.info("Setting paramters for intersections")
        self._set_params_of_each_intersection_ptcl_gen(self._ptcl_gen_params)
        log.info("Calcludating radius of intersections")
        self._compute_radius_of_intersections_for_all_tunnels()
        log.info("Separating the pointclouds of the tunnels for the intersections")
        self._separate_intersection_ptcl_from_tunnel_ptcls()
        log.info("Computing pointclouds of intersections")
        self._compute_all_intersections_ptcl()
        log.info("Computing mesh")
        self._compute_mesh()

    def _compute_mesh(self):
        points = self.complete_pointcloud
        normals = self.complete_normals
        if self._meshing_params.meshing_approach == MeshingApproaches.poisson:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self._meshing_params.poisson_depth
            )
            o3d.io.write_triangle_mesh("mesh.obj", o3d_mesh)
            self.mesh = pv.read("mesh.obj")
            os.remove("mesh.obj")
        else:
            raise NotImplementedError(
                f"The method {self._meshing_params.meshing_approach} is not implemented"
            )

    def save_mesh(self, path):
        pv.save_meshio(path, self.mesh)

    def perlin_generator_of_tunnel(
        self, tunnel: Tunnel
    ) -> CylindricalPerlinNoiseMapper:
        return self._perlin_generator_of_tunnel[tunnel]

    def multiprocesing_is_point_inside_tunnel(self, tp):
        return self.is_point_inside_tunnel(tp[0], tp[1])

    def is_point_inside_tunnel(self, tunnel: Tunnel, point, threshold=0.1):
        ad, ap, av = tunnel.spline.get_closest(point, precision=0.05)
        u, v = get_two_perpendicular_vectors(av)
        n = Point3D(point) - Point3D(ap)
        dist_to_axis = n.length
        if n.length == 0:
            return True
        n.normalize()
        proj_u = np.dot(n.xyz, u.xyz.T)
        proj_v = np.dot(n.xyz, v.xyz.T)
        angle = np.arctan2(proj_u, proj_v)
        if np.isnan(angle):
            angle = 0
        radius_of_tunnel_in_that_direction = np.linalg.norm(
            ap
            - self._compute_point_at_tunnel_by_d_and_angle(
                tunnel, ad.item(0), angle
            ).xyz
        )
        return radius_of_tunnel_in_that_direction > threshold + dist_to_axis


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
    flatten_floors,
    fta_distance,
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
    points = (
        apss
        + normals * radius
        + normals * radius * noise_magnitude * noise_to_add * pwss
    )
    if flatten_floors:
        floor_to_axis_dist = points - apss
        floor_points_idxs = np.where(floor_to_axis_dist[2] < -fta_distance)
        points[floor_points_idxs, 2] = apss[floor_points_idxs, 2] - fta_distance
    return points, normals


def points_inside_of_tunnel_section(
    axis_points, tunnel_points, points, axis_vectors=None, max_projection_over_axis=0.5
):
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


def perlin_weight_from_angle(angles, perlin_weight_angle):
    perlin_weights = np.zeros(angles.shape)
    for i, angle in enumerate(angles):
        warped_angle = warp_angle_pi(angle)
        if abs(warped_angle) < perlin_weight_angle:
            perlin_weights[i] = (abs(warped_angle) / perlin_weight_angle) ** 2
        else:
            perlin_weights[i] = 1
    return perlin_weights
