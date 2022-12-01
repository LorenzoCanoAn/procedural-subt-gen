import shapely
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import interpolate
import math
import open3d
from perlin_noise import PerlinNoise
import time
PROB_DIVERGENCE = 0.1
PROB_STOP = 0.1
MAX_SEGMENT_INCLINATION = 10/180 * math.pi  # rad
MIN_DIST_OF_MESH_POINTS = 0.1  # meters
TUNNEL_AVG_RADIUS = 3
MIN_ANGLE_FOR_INTERSECTIONS = np.deg2rad(30)
N_ANGLES_PER_CIRCLE = 10

def angles_to_vector(angles):
    th, ph = angles
    xy = math.cos(ph)
    x = xy*math.cos(th)
    y = xy*math.sin(th)
    z = math.sin(ph)
    return np.array((x, y, z))


def vector_to_angles(vector):
    x, y, z = vector
    ph = math.atan2(z, x**2+y**2)
    th = math.atan2(y, x)
    return th, ph


def warp_angle_2pi(angle):
    while angle < 0:
        angle += 2*math.pi
    return angle % (2*math.pi)


def warp_angle_pi(angle):
    new_angle = warp_angle_2pi(angle)
    if new_angle > np.pi:
        new_angle -= 2*math.pi
    return new_angle


def add_noise_to_direction(direction, horizontal_tendency, horizontal_noise, vertical_tendency, vertical_noise):
    assert direction.size == 3
    th, ph = vector_to_angles(direction)
    horizontal_deviation = np.random.normal(
        horizontal_tendency, horizontal_noise)
    th = warp_angle_2pi(th + horizontal_deviation)

    ph = np.random.normal(vertical_tendency, vertical_noise)
    if abs(ph) > MAX_SEGMENT_INCLINATION:
        ph = MAX_SEGMENT_INCLINATION * ph/abs(ph)

    direction = angles_to_vector((th, ph))
    return direction


def correct_inclination(direction):
    assert direction.size == 3
    inclination = math.asin(direction[2])
    orientation = math.atan2(direction[1], direction[0])
    if abs(inclination) > MAX_SEGMENT_INCLINATION:
        z = math.sin(MAX_SEGMENT_INCLINATION) * inclination/abs(inclination)
        x = math.cos(MAX_SEGMENT_INCLINATION) * math.cos(orientation)
        y = math.cos(MAX_SEGMENT_INCLINATION) * math.sin(orientation)
        return np.array([x, y, z])
    else:
        return direction


def correct_direction_of_intersecting_tunnel(direction, intersecting_node, angle_threshold=MIN_ANGLE_FOR_INTERSECTIONS):
    if len(intersecting_node.connected_nodes) == 0:
        return direction
    th0, ph1 = vector_to_angles(direction)
    closest_neg_angle, closest_pos_angle = np.pi, np.pi
    min_neg_difference, min_pos_difference = np.pi, np.pi
    for node in intersecting_node.connected_nodes:
        th1, ph1 = vector_to_angles(intersecting_node.xyz - node.xyz)
        difference = warp_angle_pi(th1-th0)
        if difference < 0 and abs(difference) < abs(min_neg_difference):
            min_neg_difference = difference
            closest_neg_angle = th1
        elif difference > 0 and (difference) < abs(min_pos_difference):
            min_pos_difference = difference
            closest_pos_angle = th1
    if abs(min_pos_difference) < angle_threshold and abs(min_neg_difference) < angle_threshold:
        return None
    if abs(min_neg_difference) < angle_threshold:
        thf = closest_neg_angle + angle_threshold
        return angles_to_vector((thf, ph1))
    elif abs(min_pos_difference) < angle_threshold:
        thf = closest_pos_angle - angle_threshold
        return angles_to_vector((thf, ph1))
    else:
        return direction

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
            p = np.reshape(p, [-1,1])
            u1 = np.cross(v.T, np.array([0, 1, 0]))
            u2 = np.cross(u1, v.T)
            u1 = np.reshape(u1, [-1, 1])
            u2 = np.reshape(u2, [-1, 1])
                
            angles = np.random.uniform(0, 2*math.pi, N_ANGLES_PER_CIRCLE)
            radiuses = np.array([noise([a/(2*math.pi),n/N]) for a in angles])
            normals_ = u1*np.sin(angles) + u2*np.cos(angles)
            normals_ /= np.linalg.norm(normals_, axis=0)
            
            points_ = p + normals_ * radiuses
            if not smooth_floor is None:
                indices_to_correct = (points_ - p)[-1,:]<(-smooth_floor)
                points_[-1,np.where(indices_to_correct)] = p[-1]-smooth_floor
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
                radiuses = np.array([noise([a/(2*math.pi),D]) for a in angles])
                normals_ = u1*np.sin(angles) + u2*np.cos(angles)
                normals_ /= np.linalg.norm(normals_, axis=0)
                points_ = central_point + normals_ * radiuses
                if not smooth_floor is None:
                    indices_to_correct = (points_ - central_point)[-1,:]<(-smooth_floor)
                    points_[-1,np.where(indices_to_correct)] = central_point[-1]-smooth_floor
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
                indices_to_correct = (points_ - central_point)[-1,:]<(-smooth_floor)
                points_[-1,np.where(indices_to_correct)] = central_point[-1]-smooth_floor
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
        self.noise1 = PerlinNoise(5,self.seed)
        self.noise2 = PerlinNoise(2,self.seed)
        self.noise3 = PerlinNoise(4,self.seed)
        self.noise4 = PerlinNoise(8,self.seed)

    def __call__(self, coords):
         #* self.radius/2 + self.noise2(coords) * self.radius/4 + self.noise3(coords) * self.radius/6 + self.noise4(coords) * self.radius/8
        output = self.radius + self.noise1(coords) * self.radius
        print(coords, output)
        return output
class Spline3D:
    def __init__(self, points):
        self.points = np.array(points)
        self.distances = [0 for _ in range(len(self.points))]
        for i in range(len(points)-1):
            self.distances[i+1] = self.distances[i] + np.linalg.norm(points[i+1] - points[i])
        self.distance = self.distances[-1]
        self.xspline = interpolate.splrep(self.distances,self.points[:,0])
        self.yspline = interpolate.splrep(self.distances,self.points[:,1])
        self.zspline = interpolate.splrep(self.distances,self.points[:,2])
    
    def __call__(self, d):
        assert d >= 0 and d <= self.distance
        x = interpolate.splev(d,self.xspline)
        y = interpolate.splev(d,self.yspline)
        z = interpolate.splev(d,self.zspline)
        p = np.array([x,y,z])
        x1 = interpolate.splev(d+0.001,self.xspline)
        y1 = interpolate.splev(d+0.001,self.yspline)
        z1 = interpolate.splev(d+0.001,self.zspline)
        p1 = np.array([x1,y1,z1])
        v = p1 - p
        v /= np.linalg.norm(v)
        return p, v


class Node:
    def __init__(self, coords=np.zeros(3)):
        self.connected_nodes = set()
        self.coords = coords

    def add_connection(self, node):
        self.connected_nodes.add(node)

    def remove_connection(self, node):
        self.connected_nodes.remove(node)

    @property
    def xyz(self):
        return self.coords

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]


class Edge:
    def __init__(self, n0, n1):
        self.nodes = [n0, n1]
        self.subnodes = set()
        n0.add_connection(n1)
        n1.add_connection(n0)

    def __getitem__(self, index):
        return self.nodes[index]

    def plot2d(self, ax):
        x0 = self.nodes[0].x
        x1 = self.nodes[1].x
        y0 = self.nodes[0].y
        y1 = self.nodes[1].y
        ax.plot([x0, x1], [y0, y1], c="k")

    def plot3d(self, ax):
        x0 = self.nodes[0].x
        x1 = self.nodes[1].x
        y0 = self.nodes[0].y
        y1 = self.nodes[1].y
        z0 = self.nodes[0].z
        z1 = self.nodes[1].z
        ax.plot3D([x0, x1], [y0, y1], [z0, z1], c="k")


class Tunnel:
    def __init__(self, parent):
        assert isinstance(parent, Graph)
        self.parent = parent
        self.parent.tunnels.append(self)
        # The nodes should be ordered
        self.nodes = list()
        self.distance = 0
        self._spline = None

    def split(self, node):
        assert node in self.nodes
        tunnel_1 = Tunnel(self.parent)
        tunnel_2 = Tunnel(self.parent)
        split_point = self.nodes.index(node)
        tunnel_1.set_nodes(self.nodes[:split_point+1])
        tunnel_2.set_nodes(self.nodes[split_point:])
        self.parent.add_tunnel(tunnel_1)
        self.parent.add_tunnel(tunnel_2)
        self.parent.delete_tunnel(self)

    def set_nodes(self, nodes):
        self.nodes = nodes
        self._spline = Spline3D([n.xyz for n in self.nodes])

    def add_node(self, node):
        if len(self) != 0:
            self.parent.connect_nodes(self.nodes[-1], node)
        self.nodes.append(node)
        self.parent.add_node(node)
        self._spline = None
    
    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([n.xyz for n in self.nodes])
        return self._spline

    def __len__(self):
        return len(self.nodes)


class Intersection:
    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.connected_tunnels = list()

    @property
    def n_tunnels(self):
        return len(self.connected_tunnels)


class Graph:
    def __init__(self):
        self.recalculate_control = True
        self.nodes = list()
        self.edges = list()
        self.tunnels = []  # List of lists
        self.intersections = []

    def add_node(self, node):
        self.recalculate_control = True
        if not node in self.nodes:
            self.nodes.append(node)

    def connect_nodes(self, n1, n2):
        self.recalculate_control = True
        self.edges.append(Edge(n1, n2))

    def add_floating_tunnel(self,
                            distance,
                            starting_point_coords,
                            starting_direction,
                            horizontal_tendency,
                            horizontal_noise,
                            vertical_tendency,
                            vertical_noise,
                            segment_length_avg,
                            segment_length_std):
        previous_node = Node(starting_point_coords)
        self.add_tunnel(previous_node, distance, starting_direction, horizontal_tendency,
                        horizontal_noise, vertical_tendency, vertical_noise, segment_length_avg, segment_length_std)

    def add_tunnel(self,
                   first_node,
                   distance,
                   starting_direction,
                   horizontal_tendency,
                   horizontal_noise,
                   vertical_tendency,
                   vertical_noise,
                   segment_length_avg,
                   segment_length_std):
        tunnel = Tunnel(self)
        tunnel.add_node(first_node)
        previous_orientation = correct_direction_of_intersecting_tunnel(
            starting_direction, first_node)
        previous_node = first_node
        d = 0
        while d < distance:
            # create the orientation of the segment
            segment_orientation = add_noise_to_direction(
                previous_orientation, horizontal_tendency, horizontal_noise, vertical_tendency, vertical_noise)
            segment_length = np.random.normal(
                segment_length_avg, segment_length_std)
            d += segment_length
            new_node_coords = previous_node.xyz + segment_orientation * segment_length
            new_node = Node(coords=new_node_coords)
            tunnel.add_node(new_node)
            self.connect_nodes(previous_node, new_node)
            previous_node = new_node
            previous_orientation = segment_orientation

    @property
    def minx(self):
        return min([n.x for n in self.nodes])

    @property
    def miny(self):
        return min([n.y for n in self.nodes])

    @property
    def minz(self):
        return min([n.z for n in self.nodes])

    @property
    def maxx(self):
        return max([n.x for n in self.nodes])

    @property
    def maxy(self):
        return max([n.y for n in self.nodes])

    @property
    def maxz(self):
        return max([n.z for n in self.nodes])

    def connect_with_tunnel(self, n1, n2):
        pass

    def plot2d(self):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        for edge in self.edges:
            edge.plot2d(ax)
        for node in self.nodes:
            ax.scatter(node.x, node.y, c="b")
        mincoords = np.array((self.minx, self.miny))
        maxcoords = np.array((self.maxx, self.maxy))
        max_diff = max(maxcoords-mincoords)
        ax.set_xlim(self.minx, self.minx+max_diff)
        ax.set_ylim(self.miny, self.miny+max_diff)
        plt.show()

    def plot3d(self):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        for edge in self.edges:
            edge.plot3d(ax)
        for node in self.nodes:
            ax.scatter3D(node.x, node.y, node.z, c="b")
        # for tunnel in self.tunnels:
        #    color = tuple(np.random.choice(range(256), size=3)/255)
        #    for i in range(len(tunnel)-1):
        #        x0 = tunnel[i].x
        #        x1 = tunnel[i+1].x
        #        y0 = tunnel[i].y
        #        y1 = tunnel[i+1].y
        #        z0 = tunnel[i].z
        #        z1 = tunnel[i+1].z
        #        ax.plot3D([x0, x1], [y0, y1], [z0, z1], color=color)

        #mincoords = np.array((self.minx, self.miny, self.minz))
        #maxcoords = np.array((self.maxx, self.maxy, self.maxz))
        #max_diff = max(maxcoords-mincoords)
        #ax.set_xlim(self.minx, self.minx+max_diff)
        #ax.set_ylim(self.miny, self.miny+max_diff)
        #ax.set_zlim(self.minz, self.minz+max_diff)
        plt.show()


def main():
    graph = Graph()
    graph.add_floating_tunnel(
        distance=100, starting_point_coords=np.array((0, 0, 0)),
        starting_direction=np.array((1, 0, 0)),
        horizontal_tendency=np.deg2rad(0),
        horizontal_noise=np.deg2rad(10),
        vertical_tendency=np.deg2rad(10),
        vertical_noise=np.deg2rad(5),
        segment_length_avg=10,
        segment_length_std=5)

    #graph.plot3d()
    points, normals = get_mesh_vertices_from_graph_perlin_and_spline(graph)
    ptcl = open3d.geometry.PointCloud()
    ptcl.points = open3d.utility.Vector3dVector(points.T)
    ptcl.normals = open3d.utility.Vector3dVector(normals.T)
    #mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #    ptcl, depth=10)[0]
    distances = ptcl.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    open3d.visualization.draw_geometries([ptcl])
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ptcl,depth=8, width=0, scale=1, linear_fit=True)[0]
    #vertices = np.asarray(mesh.vertices)
    #poisson_mesh.vertices = open3d.utility.Vector3dVector(
    #    vertices + np.reshape(np.random.uniform(-1, 1, vertices.size), vertices.shape))
    open3d.io.write_triangle_mesh("bpa_mesh.ply", mesh)
    


if __name__ == "__main__":
    main()
