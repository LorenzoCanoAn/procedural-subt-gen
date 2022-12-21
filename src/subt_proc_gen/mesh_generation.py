from subt_proc_gen.tunnel import Tunnel, Spline3D, CaveNode, TunnelNetwork
from subt_proc_gen.PARAMS import (
    TUNNEL_AVG_RADIUS,
    MIN_DIST_OF_MESH_POINTS,
    N_ANGLES_PER_CIRCLE,
    INTERSECTION_DISTANCE,
    HORIZONTAL_EXCLUSION_DISTANCE,
)
from subt_proc_gen.helper_functions import get_indices_of_points_below_cylinder
import math
import numpy as np
from perlin_noise import PerlinNoise
import time
import open3d as o3d
from time import time_ns as ns
import random


def get_points_along_axis(tunnel: Tunnel):
    spline = tunnel.spline
    assert isinstance(spline, Spline3D)
    # Number of circles along the spline
    N = math.ceil(spline.lenght / MIN_DIST_OF_MESH_POINTS)
    d = spline.lenght / N

    # This for loop advances through the spline circle a circle
    axis_points = None
    for n in range(N):
        p, v = spline(n * d)
        p = np.reshape(p, (1, -1))
        if axis_points is None:
            axis_points = p
        else:
            axis_points = np.vstack([axis_points, p])
    return axis_points


def get_vertices_and_normals_for_tunnel(tunnel, meshing_params):
    assert isinstance(meshing_params, TunnelMeshingParams)
    assert isinstance(tunnel, Tunnel)
    points = None
    normals = None
    centers = None
    spline = tunnel.spline
    assert isinstance(spline, Spline3D)
    noise = RadiusNoiseGenerator(spline.length, meshing_params)
    # Number of circles along the spline
    N = math.ceil(spline.length / MIN_DIST_OF_MESH_POINTS)
    d = spline.length / N
    # This for loop advances through the spline circle a circle
    for n in range(N):
        p, v = spline(n * d)
        p = np.reshape(p, [-1, 1])
        u1 = np.cross(v.T, np.array([0, 1, 0]))
        u2 = np.cross(u1, v.T)
        u1 = np.reshape(u1, [-1, 1])
        u2 = np.reshape(u2, [-1, 1])
        angles = np.random.uniform(0, 2 * math.pi, N_ANGLES_PER_CIRCLE)
        radiuses = np.array([noise([a / (2 * math.pi), n / N]) for a in angles])
        normals_ = u1 * np.sin(angles) + u2 * np.cos(angles)
        normals_ /= np.linalg.norm(normals_, axis=0)
        points_ = p + normals_ * radiuses
        # Correct the floor points so that it is flat
        if meshing_params["flatten_floor"]:
            indices_to_correct = (points_ - p)[-1, :] < (
                -meshing_params["floor_to_axis_distance"]
            )
            points_[-1, np.where(indices_to_correct)] = (
                p[-1] - meshing_params["floor_to_axis_distance"]
            )
        if points is None:
            points = points_
            normals = -normals_
            centers = np.hstack(p.T, v)
        else:
            points = np.hstack([points, points_])
            normals = np.hstack([normals, -normals_])
            centers = np.vstack([centers, p])
    return points.T, normals.T  # so the shape is Nx3


def mesh_from_vertices(points, normals):
    print("run Poisson surface reconstruction")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )

    return mesh, pcd


class RadiusNoiseGenerator:
    def __init__(self, length, meshing_params):
        assert isinstance(meshing_params, TunnelMeshingParams)
        self.radius = meshing_params["radius"]
        self.roughness = meshing_params["roughness"]
        self.seed = time.time_ns()
        self.noise1 = PerlinNoise(octaves=length * self.roughness, seed=self.seed)

    def __call__(self, coords):
        output = self.radius + self.noise1(coords) * self.radius
        return output


class TunnelMeshingParams(dict):
    def __init__(self, params=None, random=False):
        super().__init__()
        if random:
            self.random()
        else:
            self["roughness"] = 0.15
            self["flatten_floor"] = True
            self["floor_to_axis_distance"] = 1
            self["radius"] = 3

        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self[key] = params[key]

    def random(self):
        self["roughness"] = np.random.uniform(0.1, 0.2)
        self["fltatten_floor"] = random.uniform(0, 1) < 0.8
        self["floor_to_axis_distance"] = random.uniform(1, 1.5)
        self["radius"] = random.uniform(3, 3.5)


class TunnelWithMesh:
    __Tunnel_to_TunnelWithMesh = dict()

    def __init__(
        self,
        tunnel: Tunnel,
        vertices=None,
        normals=None,
        threshold_for_points_in_ends=5,
        meshing_params=TunnelMeshingParams(),
    ):
        self._tunnel = tunnel
        self.__Tunnel_to_TunnelWithMesh[
            self._tunnel
        ] = self  # This is a way to go from a tunnel to its corresponding TunnelWithMesh
        if vertices is None or normals is None:
            (
                self._raw_points,
                self._raw_normals,
                self._axis_points,
            ) = get_vertices_and_normals_for_tunnel(self._tunnel, meshing_params)
        else:
            self._raw_points, self._raw_normals = vertices, normals
        # Init the indexers
        self._indices_of_end = dict()
        self._selected_indices_of_end = dict()
        self._central_indices = np.arange(len(self._raw_points))
        self._end_indices = None
        for n, node in enumerate(self._tunnel.end_nodes):
            indices_for_this_end_node = self.get_indices_close_to_point(
                node.xyz, threshold_for_points_in_ends
            )
            if self._end_indices is None:
                self._end_indices = indices_for_this_end_node
            else:
                self._end_indices = np.concatenate(
                    (self._end_indices, indices_for_this_end_node)
                )
            self._indices_of_end[node] = np.copy(indices_for_this_end_node)
            self._selected_indices_of_end[node] = np.copy(indices_for_this_end_node)

        self._central_indices = np.delete(self._central_indices, self._end_indices)

    @classmethod
    def tunnel_to_tunnelwithmesh(cls, tunnel: Tunnel):
        tunnel_with_mesh = cls.__Tunnel_to_TunnelWithMesh[tunnel]
        assert isinstance(tunnel_with_mesh, TunnelWithMesh)
        return tunnel_with_mesh

    @property
    def tunnel(self):
        return self._tunnel

    @property
    def n_points(self):
        assert len(self._raw_normals) == len(self._raw_points)
        return len(self._raw_normals)

    # FUNCTIONS TO ACCESS THE RAW VERTICES
    @property
    def all_raw_points(self):
        return self._raw_points

    @property
    def all_raw_normals(self):
        return self._raw_normals

    @property
    def central_points(self):
        return self._raw_points[self._central_indices]

    @property
    def central_normals(self):
        return self._raw_normals[self._central_indices]

    @property
    def end_points(self):
        return self._raw_points[self._end_indices]

    @property
    def end_normals(self):
        return self._raw_normals[self._end_indices]

    @property
    def all_selected_end_indices(self):
        selected_indices = None
        for end in self._tunnel.end_nodes:
            if selected_indices is None:
                selected_indices = self._selected_indices_of_end[end]
            else:
                selected_indices = np.concatenate(
                    [selected_indices, self._selected_indices_of_end[end]]
                )
        return selected_indices

    # FUNCTIONS TO ACCESS VERTICES EXCEPT THE EXCLUDED ONES
    def selected_points_of_end(self, end_node):
        return self._raw_points[self._selected_indices_of_end[end_node]]

    @property
    def selected_end_points(self):
        return self._raw_points[self.all_selected_end_indices]

    @property
    def all_selected_indices(self):
        if len(self._central_indices) > 0:
            return np.concatenate(
                (self._central_indices, self.all_selected_end_indices)
            )
        else:
            return self.all_selected_end_indices

    @property
    def all_selected_points(self):
        return self._raw_points[self.all_selected_indices]

    @property
    def all_selected_normals(self):
        return self._raw_normals[self.all_selected_indices]

    def raw_points_in_end(self, end_node):
        assert isinstance(end_node, CaveNode)
        assert end_node in self._tunnel.end_nodes
        return self._raw_points[self._indices_of_end[end_node]]

    def deselect_point_of_end(self, end_node, to_deselect):
        self._selected_indices_of_end[end_node] = np.delete(
            self._selected_indices_of_end[end_node], to_deselect
        )

    def get_indices_close_to_point(
        self, point: np.ndarray, threshold_distance, horizontal_distance=True
    ):
        """points should have a 3x1 dimmension"""
        if horizontal_distance:
            points_xy = self.all_raw_points[:, :2]
            differences = points_xy - np.reshape(point.flatten()[:2], [1, 2])
        else:
            differences = self.all_raw_points - np.reshape(point.flatten(), [1, 3])
        distances = np.linalg.norm(differences, axis=1)
        return np.array(np.where(distances < threshold_distance)).flatten()

    def add_points_to_delete(self, indices):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        self._indices_of_excluded_vertices = np.vstack(
            self._indices_of_excluded_vertices, indices
        )


class TunnelNetworkWithMesh:
    """Wrapper around a TunnelNetwork that creates a TunnelWithMesh from each Tunnels
    and implements the intersection-cleaning functions (to remove the interior points in each of the"""

    def __init__(self, tunnel_network, meshing_params):
        assert isinstance(tunnel_network, TunnelNetwork)
        assert isinstance(meshing_params, TunnelMeshingParams)
        self._tunnel_network = tunnel_network
        self._tunnels_with_mesh = list()
        for n, tunnel in enumerate(self._tunnel_network.tunnels):
            print(
                f"Generating ptcl {n+1:>3} out of {len(tunnel_network.tunnels)}",
                end=" // ",
            )
            start = ns()
            self._tunnels_with_mesh.append(
                TunnelWithMesh(tunnel, meshing_params=meshing_params)
            )
            print(f"Time: {(ns()-start)*1e-9:<5.2f} s", end=" // ")
            print(f"{self._tunnels_with_mesh[-1].n_points:<5} points")

    def clean_intersections(self):
        n_intersections = len(self._tunnel_network.intersections)
        for n_intersection, intersection in enumerate(
            self._tunnel_network.intersections
        ):
            print(f"Cleaning intersection {n_intersection} out of {n_intersections}")
            for tunnel in intersection.tunnels:
                # Plot the central points of the tunnel
                tunnel_with_mesh_i = TunnelWithMesh.tunnel_to_tunnelwithmesh(tunnel)

                for tunnel_j in intersection.tunnels:
                    tunnel_with_mesh_j = TunnelWithMesh.tunnel_to_tunnelwithmesh(
                        tunnel_j
                    )
                    if tunnel_with_mesh_i is tunnel_with_mesh_j:
                        continue
                    # Update the secondary tunnel
                    for point in tunnel_with_mesh_i.selected_points_of_end(
                        intersection
                    ):
                        to_deselect = get_indices_of_points_below_cylinder(
                            tunnel_with_mesh_j.selected_points_of_end(intersection),
                            point,
                            HORIZONTAL_EXCLUSION_DISTANCE,
                        )
                        tunnel_with_mesh_j.deselect_point_of_end(
                            intersection, to_deselect
                        )

    def mesh_points_and_normals(self):
        mesh_points = None
        mesh_normals = None
        for n, tunnel_with_mesh in enumerate(self._tunnels_with_mesh):
            assert isinstance(tunnel_with_mesh, TunnelWithMesh)
            if mesh_points is None:
                mesh_points = tunnel_with_mesh.all_selected_points
                mesh_normals = tunnel_with_mesh.all_selected_normals
            else:
                mesh_points = np.vstack(
                    (mesh_points, tunnel_with_mesh.all_selected_points)
                )
                mesh_normals = np.vstack(
                    (mesh_normals, tunnel_with_mesh.all_selected_normals)
                )
        return mesh_points, mesh_normals
