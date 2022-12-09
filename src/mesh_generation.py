from generate_random_graph import Spline3D, Tunnel
import math
import numpy as np
from perlin_noise import PerlinNoise
import time


from PARAMS import TUNNEL_AVG_RADIUS, MIN_DIST_OF_MESH_POINTS, N_ANGLES_PER_CIRCLE

def get_mesh_vertices_from_graph_perlin_and_spline(graph, smooth_floor=1):
    points = None
    normals = None
    noise = RadiusNoiseGenerator(TUNNEL_AVG_RADIUS)
    for tunnel in graph.tunnels:
        spline = tunnel.spline
        assert isinstance(spline, Spline3D)
        N = math.ceil(spline.distance / MIN_DIST_OF_MESH_POINTS)
        d = spline.distance/N
        for n in range(N):
            p, v = spline(n*d)
            p = np.reshape(p, [-1, 1])
            u1 = np.cross(v.T, np.array([0, 1, 0]))
            u2 = np.cross(u1, v.T)
            u1 = np.reshape(u1, [-1, 1])
            u2 = np.reshape(u2, [-1, 1])

            angles = np.random.uniform(0, 2*math.pi, N_ANGLES_PER_CIRCLE)
            radiuses = np.array([noise([a/(2*math.pi), n/N]) for a in angles])
            normals_ = u1*np.sin(angles) + u2*np.cos(angles)
            normals_ /= np.linalg.norm(normals_, axis=0)

            points_ = p + normals_ * radiuses
            if not smooth_floor is None:
                indices_to_correct = (points_ - p)[-1, :] < (-smooth_floor)
                points_[-1, np.where(indices_to_correct)] = p[-1]-smooth_floor
            if points is None:
                points = points_
                normals = -normals_
            else:
                points = np.hstack([points, points_])
                normals = np.hstack([normals, -normals_])

        return points, normals


def get_mesh_vertices_from_graph_perlin(graph, smooth_floor=1):
    points = None
    normals = None
    noise = RadiusNoiseGenerator(TUNNEL_AVG_RADIUS)
    for tunnel in graph.tunnels:
        assert isinstance(tunnel, Tunnel)
        tunnel.spline
        D = 0
        for i in range(len(tunnel.nodes)-1):
            p0 = tunnel.nodes[i].xyz
            p1 = tunnel.nodes[i+1].xyz
            seg = p1-p0
            seg_d = np.linalg.norm(seg)
            dir = seg / seg_d
            n = math.ceil(seg_d/MIN_DIST_OF_MESH_POINTS)
            d = seg_d/n
            D += d
            v = dir*d
            u1 = np.cross(dir, np.array([0, 1, 0]))
            u2 = np.cross(u1, dir)
            u1 = np.reshape(u1, [-1, 1])
            u2 = np.reshape(u2, [-1, 1])
            for i in range(1, n+1):
                central_point = p0 + v*i
                central_point = np.reshape(central_point, [-1, 1])
                angles = np.random.uniform(0, 2*math.pi, N_ANGLES_PER_CIRCLE)
                radiuses = np.array([noise([a/(2*math.pi), D])
                                    for a in angles])
                normals_ = u1*np.sin(angles) + u2*np.cos(angles)
                normals_ /= np.linalg.norm(normals_, axis=0)
                points_ = central_point + normals_ * radiuses
                if not smooth_floor is None:
                    indices_to_correct = (
                        points_ - central_point)[-1, :] < (-smooth_floor)
                    points_[-1, np.where(indices_to_correct)
                            ] = central_point[-1]-smooth_floor
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
        seg = p1-p0
        seg_d = np.linalg.norm(seg)
        dir = seg / seg_d
        n = math.ceil(seg_d/MIN_DIST_OF_MESH_POINTS)
        d = seg_d/n
        v = dir*d
        u1 = np.cross(dir, np.array([0, 1, 0]))
        u2 = np.cross(u1, dir)
        u1 = np.reshape(u1, [-1, 1])
        u2 = np.reshape(u2, [-1, 1])
        for i in range(1, n+1):
            central_point = p0 + v*i
            central_point = np.reshape(central_point, [-1, 1])
            angles = np.random.uniform(0, 2*math.pi, 20)
            normals_ = u1*np.sin(angles) + u2*np.cos(angles)
            normals_ /= np.linalg.norm(normals_, axis=0)
            points_ = central_point + normals_ * \
                np.random.normal(TUNNEL_AVG_RADIUS, 0)
            if not smooth_floor is None:
                indices_to_correct = (
                    points_ - central_point)[-1, :] < (-smooth_floor)
                points_[-1, np.where(indices_to_correct)
                        ] = central_point[-1]-smooth_floor
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
        print(coords, output)
        return output