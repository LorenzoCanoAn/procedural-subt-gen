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
        p = np.reshape(p, (-1, 1))
        if axis_points is None:
            axis_points = p
        else:
            axis_points = np.hstack([axis_points, p])
    ptcl = o3d.geometry.PointCloud()
    print(axis_points.shape)
    ptcl.points = o3d.utility.Vector3dVector(axis_points.T)
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
    return points, normals


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
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.normals = o3d.utility.Vector3dVector(normals.T)
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
    def __init__(self, tunnel: Tunnel, vertices=None, normals=None):
        self._tunnel = tunnel
        if vertices is None or normals is None:
            self._points, self._normals = get_vertices_and_normals_for_tunnel(tunnel)
        else:
            self._points, self._normals = vertices, normals

        self._indices_of_excluded_vertices = np.array([], dtype=np.int32)
        self._ptcl = None

    @property
    def tunnel(self):
        return self._tunnel

    @property
    def ptcl(self):
        if self._ptcl is None:
            self.gen_ptcl()
        return self._ptcl

    @property
    def points(self):
        return self._points

    @property
    def filtered_points(self):
        return np.delete(self._points, self._indices_of_excluded_vertices)

    @property
    def filtered_normals(self):
        return np.delete(self._normals, self._indices_of_excluded_vertices)

    @property
    def normals(self):
        return self._normals

    @property
    def n_points(self):
        assert len(self._normals.T) == len(self._points.T)
        return len(self._normals.T)

    @property
    def n_filtered_points(self):
        n_filtered_normals = len(self.filtered_normals.T)
        n_filtered_points = len(self.filtered_points.T)
        assert n_filtered_points == n_filtered_normals
        return n_filtered_normals

    def get_points_close_to_point(self, point, threshold_distance):
        """points should have a 3x1 dimmension"""
        return np.where(np.linalg.norm(self.points - np.reshape(point, (-1, 1)), axis=0) < threshold_distance)

    def gen_ptcl(self):
        self._ptcl = o3d.geometry.PointCloud()
        self._ptcl.points = o3d.utility.Vector3dVector(self._points.T)
        self._ptcl.normals = o3d.utility.Vector3dVector(self._normals.T)

    def add_points_to_delete(self, indices):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        self._indices_of_excluded_vertices = np.vstack(
            self._indices_of_excluded_vertices, indices
        )

    def get_points_to_delete_from_other_tunnel(self, other_tunnel, threshold_distance = 4):
        assert isinstance(other_tunnel, TunnelWithMesh)
        nodes_to_check = set()  # Nodes where the two tunnels coincide
        for node in other_tunnel.tunnel.nodes:
            if node in self.tunnel._nodes:
                nodes_to_check.add(node)

        for node_to_check in nodes_to_check:
            print(f"{self} will check with {other_tunnel} in node {node_to_check}")
            own_points_indices = self.get_points_close_to_point(node.xyz,threshold_distance=threshold_distance)
            return own_points_indices, node_to_check
        return None, None
def o3d_to_mshlib_mesh(mesh):
    pass
