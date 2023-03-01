import os
from subt_proc_gen.tunnel import Tunnel, Spline3D, CaveNode, TunnelNetwork
from subt_proc_gen.PARAMS import (
    TUNNEL_AVG_RADIUS,
    MIN_DIST_OF_MESH_POINTS,
    N_ANGLES_PER_CIRCLE,
    INTERSECTION_DISTANCE,
)
from subt_proc_gen.helper_functions import (
    get_indices_close_to_point,
    get_two_perpendicular_vectors_to_vector,
)
import math
import numpy as np
from perlin_noise import PerlinNoise
import open3d as o3d
from time import time_ns as ns
import random


def get_mesh_points_of_tunnel(tunnel, meshing_params):
    assert isinstance(meshing_params, TunnelMeshingParams)
    assert isinstance(tunnel, Tunnel)
    assert isinstance(tunnel.spline, Spline3D)
    spline = tunnel.spline
    noise = TunnelNoiseGenerator(spline.length, meshing_params)
    # Number of circles along the spline
    N = math.ceil(spline.length / MIN_DIST_OF_MESH_POINTS)
    d = spline.length / N
    n_a = N_ANGLES_PER_CIRCLE
    points = np.zeros([N * n_a, 3])
    normals = np.zeros([N * n_a, 3])
    # axis points and vectors
    ads, aps, avs = tunnel.spline.discretized
    # This for loop advances through the spline circle a circle
    for n in range(N):
        ap, av = aps[n], avs[n]  # axis point and vector
        # u1 and u2 are perpendicular to av. To get u1, you do the cross-product of av
        # with a non paralel vector
        angles = np.linspace(0, 2 * math.pi, n_a).reshape([-1, 1])
        radiuses = np.array([noise(n / N, a) for a in angles]).reshape([-1, 1])
        u1, u2 = get_two_perpendicular_vectors_to_vector(av)
        normals_ = -(u1 * np.sin(angles) + u2 * np.cos(angles))
        normals_ /= np.linalg.norm(normals_, axis=1).reshape([-1, 1])
        points_ = ap + normals_ * radiuses
        start = n * n_a
        stop = n * n_a + n_a
        points[start:stop] = points_
        normals[start:stop] = -normals_
    extended_centers = (np.ones((N, n_a, 3)) * aps.reshape([-1, 1, 3])).reshape(
        n_a * N, 3
    )
    if meshing_params["flatten_floor"]:
        indices_to_correct = np.where(
            (points - extended_centers)[:, -1]
            < (-meshing_params["floor_to_axis_distance"])
        )
        floor_points = points[indices_to_correct, :][0]
        floor_normals = normals[indices_to_correct, :][0]
        points = np.delete(points, indices_to_correct, 0)
        normals = np.delete(normals, indices_to_correct, 0)
        floor_points[:, -1] = (
            extended_centers[indices_to_correct, -1]
            - meshing_params["floor_to_axis_distance"]
        )
        floor_normals[:, :] = np.array([0, 0, 1]).reshape(1, 3)
    else:
        floor_points = np.zeros((0, 3))
        floor_normals = np.zeros((0, 3))
    return points, normals, floor_points, floor_normals, noise  # so the shape is Nx3


def mesh_from_vertices(points, normals, method, poisson_depth=11):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    if method == "poisson":
        pcd.normals = o3d.utility.Vector3dVector(normals)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
    elif method == "ball":
        radii = [0.6, 0.5, 0.4, 0.3]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    return mesh, pcd


class TunnelNoiseGenerator:
    def __init__(self, length, meshing_params):
        assert isinstance(meshing_params, TunnelMeshingParams)
        self.lenght = length
        self.radius = meshing_params["radius"]
        self.roughness = meshing_params["roughness"]
        self.seed = 1677436382510714946  # time.time_ns()
        if self.roughness != 0.0:
            self.noise1 = PerlinNoise(octaves=length * self.roughness, seed=self.seed)
            self.noise2 = PerlinNoise(
                octaves=length * self.roughness * 2, seed=self.seed
            )
            self.noise3 = PerlinNoise(
                octaves=length * self.roughness * 4, seed=self.seed
            )

    def __call__(self, d, angle):
        if self.roughness == 0:
            return self.radius
        l = angle * self.radius / self.lenght
        n1 = self.noise1((d, l))
        n2 = self.noise2((d, l))
        n3 = self.noise3((d, l))
        output = (
            self.radius + n1 * self.radius + n2 * self.radius / 2 + n3 * self.radius / 4
        )
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
            self["radius"] = TUNNEL_AVG_RADIUS

        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self[key] = params[key]

    def random(self):
        self["roughness"] = np.random.uniform(0.3)
        self["flatten_floor"] = True
        self["floor_to_axis_distance"] = random.uniform(1, 1.5)
        self["radius"] = TUNNEL_AVG_RADIUS


class TunnelWithMesh:
    __Tunnel_to_TunnelWithMesh = dict()

    def __init__(
        self,
        tunnel: Tunnel,
        p2i_dist=INTERSECTION_DISTANCE,
        meshing_params=TunnelMeshingParams(),
    ):
        self.p2i_dist = p2i_dist
        self._tunnel = tunnel
        self.__Tunnel_to_TunnelWithMesh[
            self._tunnel
        ] = self  # This is a way to go from a tunnel to its corresponding TunnelWithMesh
        (
            self._raw_points,
            self._raw_normals,
            self._floor_points,
            self._floor_normals,
            self._noise,
        ) = get_mesh_points_of_tunnel(self._tunnel, meshing_params)
        self.n_points = self._raw_points.shape[0]
        # Init the indexers
        self.points_at_intersection = dict()
        self.normals_at_intersection = dict()
        self.central_points = np.copy(self._raw_points)
        self.central_normals = np.copy(self._raw_normals)

    def get_xy_projection(self, precission=2):

        spline = self._tunnel.spline
        noise = self._noise
        # Number of circles along the spline
        N = math.ceil(spline.length / precission)
        n_a = 2
        points = np.zeros([N * n_a, 3])
        normals = np.zeros([N * n_a, 3])
        # axis points and vectors
        ads, aps, avs = spline.discretize(precission)
        # This for loop advances through the spline circle a circle
        for n in range(N):
            ap, av = aps[n], avs[n]  # axis point and vector
            # u1 and u2 are perpendicular to av. To get u1, you do the cross-product of av
            # with a non paralel vector
            angles = np.array([0, np.pi]).reshape([-1, 1])
            radiuses = np.array([noise(n / N, a) for a in angles]).reshape([-1, 1])
            u1, u2 = get_two_perpendicular_vectors_to_vector(av)
            normals_ = u1 * np.cos(angles) + u2 * np.sin(angles)
            normals_ /= np.linalg.norm(normals_, axis=1).reshape([-1, 1])
            points_ = ap + normals_ * radiuses
            start = n * n_a
            stop = n * n_a + n_a
            points[start:stop] = points_
            normals[start:stop] = -normals_
        return points[:, :2], normals[:, :2]

    def add_intersection(self, node):
        indices_for_this_end_node = get_indices_close_to_point(
            self.central_points, node.xyz, self.p2i_dist
        )
        self.points_at_intersection[node] = self.central_points[
            indices_for_this_end_node, :
        ]
        self.normals_at_intersection[node] = self.central_normals[
            indices_for_this_end_node, :
        ]
        self.central_points = np.delete(
            self.central_points, indices_for_this_end_node, axis=0
        )
        self.central_normals = np.delete(
            self.central_normals, indices_for_this_end_node, axis=0
        )

    def delete_points_in_end(self, end_node, indices):
        self.points_at_intersection[end_node] = np.delete(
            self.points_at_intersection[end_node], indices, axis=0
        )
        self.normals_at_intersection[end_node] = np.delete(
            self.normals_at_intersection[end_node], indices, axis=0
        )

    @classmethod
    def tunnel_to_tunnelwithmesh(cls, tunnel: Tunnel):
        tunnel_with_mesh = cls.__Tunnel_to_TunnelWithMesh[tunnel]
        assert isinstance(tunnel_with_mesh, TunnelWithMesh)
        return tunnel_with_mesh

    def is_point_inside(self, point):
        spline = self._tunnel.spline
        d_ap_av = spline.get_most_perpendicular_point_in_spline(point, 5)
        if d_ap_av is None:
            return False
        else:
            d, ap, av = d_ap_av
            ap = np.reshape(ap, [1, -1])
            av = np.reshape(av, [1, -1])
        vap_p = point - ap  # vector form ap to p
        p_r = np.linalg.norm(vap_p)
        n = vap_p / p_r  # normal of p
        u1, u2 = get_two_perpendicular_vectors_to_vector(av)
        nx = n[0, 0]
        ny = n[0, 1]
        u1x = u1[0, 0]
        u1y = u1[0, 1]
        u2x = u2[0, 0]
        u2y = u2[0, 1]
        a = np.arctan2(ny * u2y - nx * u2x, nx * u1x - ny * u1y)
        t_r = self._noise(d, a)
        is_inside = t_r > p_r
        return is_inside

    @property
    def all_selected_points(self):
        selected_points = self.central_points
        for node in self.points_at_intersection.keys():
            selected_points = np.concatenate(
                [selected_points, self.points_at_intersection[node]]
            )
        selected_points = np.concatenate([selected_points, self._floor_points])
        return selected_points

    @property
    def all_selected_normals(self):
        selected_normals = self.central_normals
        for node in self.normals_at_intersection.keys():
            selected_normals = np.concatenate(
                [selected_normals, self.normals_at_intersection[node]]
            )
        selected_normals = np.concatenate([selected_normals, self._floor_normals])
        return selected_normals


class TunnelNetworkWithMesh:
    """Wrapper around a TunnelNetwork that creates a TunnelWithMesh from each Tunnels
    and implements the intersection-cleaning functions (to remove the interior points in each of the"""

    def __init__(self, tunnel_network: TunnelNetwork, i_meshing_params):
        assert isinstance(tunnel_network, TunnelNetwork)
        self._tunnel_network = tunnel_network
        self._tunnels_with_mesh = list()
        for n, tunnel in enumerate(self._tunnel_network.tunnels):
            print(
                f"Generating ptcl {n+1:>3} out of {len(tunnel_network.tunnels)}",
                end=" // ",
            )
            start = ns()
            if i_meshing_params == "random":
                meshing_params = TunnelMeshingParams(random=True)
            else:
                assert isinstance(i_meshing_params, TunnelMeshingParams)
                meshing_params = i_meshing_params
            self._tunnels_with_mesh.append(
                TunnelWithMesh(tunnel, meshing_params=meshing_params)
            )
            print(f"Time: {(ns()-start)*1e-9:<5.2f} s", end=" // ")
            print(f"{self._tunnels_with_mesh[-1].n_points:<5} points")
        for intersection in self._tunnel_network.intersections:
            for tunnel in intersection.tunnels:
                ti = TunnelWithMesh.tunnel_to_tunnelwithmesh(tunnel).add_intersection(
                    intersection
                )

    def clean_intersections(self):
        n_intersections = len(self._tunnel_network.intersections)
        for n_intersection, intersection in enumerate(
            self._tunnel_network.intersections
        ):
            for tnmi in intersection.tunnels:
                for tnmj in intersection.tunnels:
                    assert isinstance(tnmj, Tunnel)
                    assert isinstance(tnmi, Tunnel)
                    if tnmi is tnmj:
                        continue
                    ti = TunnelWithMesh.tunnel_to_tunnelwithmesh(tnmi)
                    tj = TunnelWithMesh.tunnel_to_tunnelwithmesh(tnmj)
                    assert isinstance(ti, TunnelWithMesh)
                    assert isinstance(tj, TunnelWithMesh)
                    indices_to_delete = []
                    pis = ti.points_at_intersection[intersection]
                    for npi, pi in enumerate(pis):

                        if tj.is_point_inside(pi):
                            indices_to_delete.append(npi)
                    ti.delete_points_in_end(intersection, indices_to_delete)

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

    def compute_mesh(self):
        points, normals = self.mesh_points_and_normals()
        mesh, ptcl = mesh_from_vertices(
            points, normals, method="poisson", poisson_depth=11
        )
        simplified_mesh = mesh.simplify_quadric_decimation(
            int(len(mesh.triangles) * 0.3)
        )
        print(f"Original mesh has {len(mesh.triangles)} triangles")
        print(f"Simplified mesh has {len(simplified_mesh.triangles)} triangles")
        self.mesh = mesh
        self.simplified_mesh = simplified_mesh
        return mesh, simplified_mesh

    def save_mesh(self, file_path):
        root, extension = os.path.splitext(file_path)
        simplified_mesh_file_path = root + "_simplified" + extension
        o3d.io.write_triangle_mesh(file_path, self.mesh)
        o3d.io.write_triangle_mesh(simplified_mesh_file_path, self.simplified_mesh)


def generate_mesh_from_tunnel_network(tunnel_network: TunnelNetwork, path_to_mesh):
    # Order the tunnels so that the meshes intersect
    tunnel_network_with_mesh = TunnelNetworkWithMesh(
        tunnel_network, meshing_params=TunnelMeshingParams({"roughness": 0.1})
    )
    tunnel_network_with_mesh.clean_intersections()
    points, normals = tunnel_network_with_mesh.mesh_points_and_normals()
    save_folder, filename = os.path.split(path_to_mesh)
    np.save(os.path.join(save_folder, "points.np"), points)
    np.save(os.path.join(save_folder, "normals.np"), normals)

    mesh, ptcl = mesh_from_vertices(points, normals, method="poisson", poisson_depth=11)
    simplified_mesh = mesh.simplify_quadric_decimation(int(len(mesh.triangles) * 0.3))
    print(f"Original mesh has {len(mesh.triangles)} triangles")
    print(f"Simplified mesh has {len(simplified_mesh.triangles)} triangles")
    print(f"Writing mesh to : {path_to_mesh}")
    o3d.io.write_triangle_mesh(
        path_to_mesh,
        simplified_mesh,
    )
