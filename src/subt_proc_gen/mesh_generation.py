from subt_proc_gen.tunnel import Tunnel, Spline3D
import math
import numpy as np
from perlin_noise import PerlinNoise
import time
import open3d as o3d

from subt_proc_gen.PARAMS import (
    TUNNEL_AVG_RADIUS,
    MIN_DIST_OF_MESH_POINTS,
    N_ANGLES_PER_CIRCLE,
)


def get_axis_pointcloud(tunnel: Tunnel):
    spline = tunnel.spline
    assert isinstance(spline, Spline3D)
    # Number of circles along the spline
    N = math.ceil(spline.distance / MIN_DIST_OF_MESH_POINTS)
    d = spline.distance / N

    # This for loop advances through the spline circle a circle
    axis_points = None
    for n in range(N):
        p, v = spline(n * d)
        p = np.reshape(p, (1, -1))
        if axis_points is None:
            axis_points = p
        else:
            axis_points = np.vstack([axis_points, p])
    ptcl = o3d.geometry.PointCloud()
    print(axis_points.shape)
    ptcl.points = o3d.utility.Vector3dVector(axis_points)
    ptcl.colors = o3d.utility.Vector3dVector(
        np.ones(np.asarray(ptcl.points).shape) * np.array((0, 0, 0))
    )
    return ptcl


def get_vertices_and_normals_for_tunnel(tunnel, smooth_floor=1):
    noise = RadiusNoiseGenerator(TUNNEL_AVG_RADIUS)
    points = None
    normals = None
    spline = tunnel.spline
    assert isinstance(spline, Spline3D)
    # Number of circles along the spline
    N = math.ceil(spline.distance / MIN_DIST_OF_MESH_POINTS)
    d = spline.distance / N

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
        if not smooth_floor is None:
            indices_to_correct = (points_ - p)[-1, :] < (-smooth_floor)
            points_[-1, np.where(indices_to_correct)] = p[-1] - smooth_floor
        if points is None:
            points = points_
            normals = -normals_
        else:
            points = np.hstack([points, points_])
            normals = np.hstack([normals, -normals_])
    return points.T, normals.T  # so the shape is Nx3


def get_vertices_for_tunnels(graph, smooth_floor=1):
    tunnels_points = list()
    tunnels_normals = list()
    for tunnel in graph._tunnels:
        points, normals = get_vertices_and_normals_for_tunnel(tunnel, smooth_floor)

        tunnels_points.append(points)
        tunnels_normals.append(normals)
    return tunnels_points, tunnels_normals


def get_mesh_vertices_from_graph_perlin_and_spline(graph, smooth_floor=1):
    points = None
    normals = None
    noise = RadiusNoiseGenerator(TUNNEL_AVG_RADIUS)
    for tunnel in graph._tunnels:
        spline = tunnel.spline
        assert isinstance(spline, Spline3D)
        # Number of circles along the spline
        N = math.ceil(spline.distance / MIN_DIST_OF_MESH_POINTS)
        d = spline.distance / N

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
            if not smooth_floor is None:
                indices_to_correct = (points_ - p)[-1, :] < (-smooth_floor)
                points_[-1, np.where(indices_to_correct)] = p[-1] - smooth_floor

            if points is None:
                points = points_
                normals = -normals_
            else:
                points = np.hstack([points, points_])
                normals = np.hstack([normals, -normals_])

    return points, normals


def mesh_from_vertices(points, normals):
    print("run Poisson surface reconstruction")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=7
    )
    return mesh, pcd


def plot_mesh(mesh):
    o3d.visualization.draw_geometries(
        [mesh],
        zoom=0.664,
        front=[-0.4761, -0.4698, -0.7434],
        lookat=[1.8900, 3.2596, 0.9284],
        up=[0.2304, -0.8825, 0.4101],
    )


def get_mesh_vertices_from_graph_perlin(graph, smooth_floor=1):
    points = None
    normals = None
    noise = RadiusNoiseGenerator(TUNNEL_AVG_RADIUS)
    for tunnel in graph.tunnels:
        assert isinstance(tunnel, Tunnel)
        tunnel.spline
        D = 0
        for i in range(len(tunnel.nodes) - 1):
            p0 = tunnel.nodes[i].xyz
            p1 = tunnel.nodes[i + 1].xyz
            seg = p1 - p0
            seg_d = np.linalg.norm(seg)
            dir = seg / seg_d
            n = math.ceil(seg_d / MIN_DIST_OF_MESH_POINTS)
            d = seg_d / n
            D += d
            v = dir * d
            u1 = np.cross(dir, np.array([0, 1, 0]))
            u2 = np.cross(u1, dir)
            u1 = np.reshape(u1, [-1, 1])
            u2 = np.reshape(u2, [-1, 1])
            for i in range(1, n + 1):
                central_point = p0 + v * i
                central_point = np.reshape(central_point, [-1, 1])
                angles = np.random.uniform(0, 2 * math.pi, N_ANGLES_PER_CIRCLE)
                radiuses = np.array([noise([a / (2 * math.pi), D]) for a in angles])
                normals_ = u1 * np.sin(angles) + u2 * np.cos(angles)
                normals_ /= np.linalg.norm(normals_, axis=0)
                points_ = central_point + normals_ * radiuses
                if not smooth_floor is None:
                    indices_to_correct = (points_ - central_point)[-1, :] < (
                        -smooth_floor
                    )
                    points_[-1, np.where(indices_to_correct)] = (
                        central_point[-1] - smooth_floor
                    )
                if points is None:
                    points = points_
                    normals = -normals_
                else:
                    points = np.hstack([points, points_])
                    normals = np.hstack([normals, -normals_])
        return points, normals


def get_mesh_vertices_from_graph(graph, smooth_floor=1):
    points = None
    normals = None
    for edge in graph.edges:
        p0 = edge[0].xyz
        p1 = edge[1].xyz
        seg = p1 - p0
        seg_d = np.linalg.norm(seg)
        dir = seg / seg_d
        n = math.ceil(seg_d / MIN_DIST_OF_MESH_POINTS)
        d = seg_d / n
        v = dir * d
        u1 = np.cross(dir, np.array([0, 1, 0]))
        u2 = np.cross(u1, dir)
        u1 = np.reshape(u1, [-1, 1])
        u2 = np.reshape(u2, [-1, 1])
        for i in range(1, n + 1):
            central_point = p0 + v * i
            central_point = np.reshape(central_point, [-1, 1])
            angles = np.random.uniform(0, 2 * math.pi, 20)
            normals_ = u1 * np.sin(angles) + u2 * np.cos(angles)
            normals_ /= np.linalg.norm(normals_, axis=0)
            points_ = central_point + normals_ * np.random.normal(TUNNEL_AVG_RADIUS, 0)
            if not smooth_floor is None:
                indices_to_correct = (points_ - central_point)[-1, :] < (-smooth_floor)
                points_[-1, np.where(indices_to_correct)] = (
                    central_point[-1] - smooth_floor
                )
            if points is None:
                points = points_
                normals = -normals_
            else:
                points = np.hstack([points, points_])
                normals = np.hstack([normals, -normals_])
    return points, normals


class RadiusNoiseGenerator:
    def __init__(self, radius):
        self.radius = radius
        self.seed = time.time_ns()
        self.noise1 = PerlinNoise(5, self.seed)
        self.noise2 = PerlinNoise(2, self.seed)
        self.noise3 = PerlinNoise(4, self.seed)
        self.noise4 = PerlinNoise(8, self.seed)

    def __call__(self, coords):
        # * self.radius/2 + self.noise2(coords) * self.radius/4 + self.noise3(coords) * self.radius/6 + self.noise4(coords) * self.radius/8
        output = self.radius + self.noise1(coords) * self.radius
        return output


class TunnelWithMesh:
    Tunnel_to_TunnelWithMesh = dict()

    def __init__(
        self,
        tunnel: Tunnel,
        vertices=None,
        normals=None,
        threshold_for_points_in_ends=5,
    ):
        self._tunnel = tunnel
        self.Tunnel_to_TunnelWithMesh[
            self._tunnel
        ] = self  # This is a way to go from a tunnel to its corresponding TunnelWithMesh
        if vertices is None or normals is None:
            self._raw_points, self._raw_normals = get_vertices_and_normals_for_tunnel(
                tunnel
            )
        else:
            self._raw_points, self._raw_normals = vertices, normals

        self._indices_of_excluded_vertices = np.array([], dtype=np.int32)
        self._ptcl = None
        self._indices_in_ends = {
            self.tunnel.nodes[0]: self.get_indices_close_to_point(
                self.tunnel.nodes[0].xyz, threshold_for_points_in_ends
            ),
            self.tunnel.nodes[-1]: self.get_indices_close_to_point(
                self.tunnel.nodes[-1].xyz, threshold_for_points_in_ends
            ),
        }

        self._filtered_indices_in_ends = {
            self.tunnel.nodes[0]: np.array(()),
            self.tunnel.nodes[-1]: np.array(()),
        }

    @property
    def tunnel(self):
        return self._tunnel

    @property
    def ptcl(self):
        if self._ptcl is None:
            self.gen_ptcl()
        return self._ptcl

    @property
    def raw_points(self):
        return self._raw_points

    @property
    def raw_normals(self):
        return self._raw_normals

    @property
    def filtered_points(self):
        raise NotImplementedError()

    @property
    def filtered_normals(self):
        raise NotImplementedError()

    @property
    def filtered_points_and_normals(self):
        return self.filtered_points, self.filtered_normals

    @property
    def n_points(self):
        assert len(self._raw_normals) == len(self._raw_points)
        return len(self._raw_normals)

    def get_indices_close_to_point(
        self, point: np.ndarray, threshold_distance, horizontal_distance=True
    ):
        """points should have a 3x1 dimmension"""
        if horizontal_distance:
            points_xy = self.raw_points[:, :2]
            differences = points_xy - np.reshape(point.flatten()[:2], [1, -1])
        else:
            differences = self.raw_points - np.reshape(point.flatten(), [1, -1])
        distances = np.linalg.norm(differences, axis=1)
        return np.where(distances < threshold_distance)

    def gen_ptcl(self):
        self._ptcl = o3d.geometry.PointCloud()
        self._ptcl.points = o3d.utility.Vector3dVector(self._raw_points)
        self._ptcl.normals = o3d.utility.Vector3dVector(self._raw_normals)

    def add_points_to_delete(self, indices):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        self._indices_of_excluded_vertices = np.vstack(
            self._indices_of_excluded_vertices, indices
        )


def o3d_to_mshlib_mesh(mesh):
    pass
