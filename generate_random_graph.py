import shapely
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import open3d

PROB_DIVERGENCE = 0.1
PROB_STOP = 0.1
MAX_SEGMENT_INCLINATION = 10/180 * math.pi  # rad
MIN_DIST_OF_MESH_POINTS = 1  # meters
TUNNEL_AVG_RADIUS = 3


def add_noise_to_direction(direction):
    assert direction.size == 3
    direction += np.random.normal(0, 0.5, 3)
    direction /= np.linalg.norm(direction)
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


def get_mesh_from_graph(graph):
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
        for i in range(n+1):
            central_point = p0 + v*i
            central_point = np.reshape(central_point, [-1, 1])
            angles = np.random.uniform(0, 2*math.pi, 5)
            normals_ = u1*np.sin(angles) + u2*np.cos(angles)
            normals_ /= np.linalg.norm(normals_, axis=0)
            if points is None:
                points = central_point + normals_*TUNNEL_AVG_RADIUS
                normals = -normals_
            else:
                points = np.hstack(
                    [points, central_point+normals_*TUNNEL_AVG_RADIUS])
                normals = np.hstack([normals, -normals_])
    return points, normals


class Node:
    def __init__(self, coords=np.zeros(3)):
        self.connected_nodes = set()
        self.coords = coords

    def add_connection(self, node):
        self.connected_nodes.add(node)

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

    def plot(self, ax):
        x0 = self.nodes[0].x
        x1 = self.nodes[1].x
        y0 = self.nodes[0].y
        y1 = self.nodes[1].y
        z0 = self.nodes[0].z
        z1 = self.nodes[1].z
        ax.plot3D([x0, x1], [y0, y1], [z0, z1], c="k")


class Graph:
    def __init__(self):
        self.recalculate_control = True
        self.nodes = list()
        self.edges = list()
        self._tunnels = []  # List of lists
        self.intersections = []

    def add_node(self, node):
        self.recalculate_control = True
        self.nodes.append(node)

    def connect_nodes(self, n1, n2):
        self.recalculate_control = True
        self.edges.append(Edge(n1, n2))

    def add_tunnel(self, distance=None, base_node=None, direction=None):
        if base_node is None:
            base_node = random.choice(self.nodes)
        if direction is None:
            direction = np.random.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
        if distance is None:
            distance = np.random.normal(100, 50)
        d = 0
        previous_orientation = direction
        previous_node = base_node
        while d < distance:
            # create the orientation of the segment
            segment_orientation = add_noise_to_direction(previous_orientation)
            segment_orientation = correct_inclination(segment_orientation)
            segment_distance = max(np.random.normal(10, 5), 5)
            new_node_coords = previous_node.xyz + segment_orientation*segment_distance
            d += segment_distance
            new_node = Node(coords=new_node_coords)
            self.add_node(new_node)
            self.connect_nodes(previous_node, new_node)
            previous_node = new_node
            previous_orientation = segment_orientation

    def recalculate(self):
        self._intersection_nodes = list()
        tunnel_nodes = list()
        self._tunnels = list()
        for node in self.nodes:
            assert isinstance(node, Node)
            if len(node.connected_nodes) > 2:
                self._intersection_nodes.append(node)
            else:
                tunnel_nodes.append(node)
        # Get the tunnels
        for intersection in self._intersection_nodes:
            for first_tunnel_node in intersection.connected_nodes:
                tunnel = list()
                tunnel_already_done = False
                # this is the start of a tunnel
                if first_tunnel_node in self._intersection_nodes:
                    continue
                else:
                    for t in self._tunnels:
                        if first_tunnel_node in t:
                            tunnel_already_done = True
                    if tunnel_already_done:
                        continue
                    tunnel.append(first_tunnel_node)
                    prev_tunnel_node = first_tunnel_node
                    reached_end_of_tunnel = False
                    while not reached_end_of_tunnel:
                        if len(prev_tunnel_node.connected_nodes) == 1:
                            break
                        for connection in prev_tunnel_node.connected_nodes:
                            if not connection in tunnel:
                                if not connection in self._intersection_nodes:
                                    tunnel.append(connection)
                                    prev_tunnel_node = connection
                                    continue
                                else:
                                    if not connection is intersection:
                                        reached_end_of_tunnel = True
                self._tunnels.append(tunnel)
        self.recalculate_control = False

    @property
    def intersection_nodes(self):
        if self.recalculate_control:
            self.recalculate()
        return self._intersection_nodes

    @property
    def tunnels(self):
        if self.recalculate_control:
            self.recalculate()
        return self._tunnels

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

    def plot(self):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        for edge in self.edges:
            edge.plot(ax)
        for node in self.intersection_nodes:
            ax.scatter3D(node.x, node.y, node.z, c="b")
        for tunnel in self.tunnels:
            color = tuple(np.random.choice(range(256), size=3)/255)
            for i in range(len(tunnel)-1):
                x0 = tunnel[i].x
                x1 = tunnel[i+1].x
                y0 = tunnel[i].y
                y1 = tunnel[i+1].y
                z0 = tunnel[i].z
                z1 = tunnel[i+1].z
                ax.plot3D([x0, x1], [y0, y1], [z0, z1], color=color)

        mincoords = np.array((self.minx, self.miny, self.minz))
        maxcoords = np.array((self.maxx, self.maxy, self.maxz))
        max_diff = max(maxcoords-mincoords)
        ax.set_xlim(self.minx, self.minx+max_diff)
        ax.set_ylim(self.miny, self.miny+max_diff)
        ax.set_zlim(self.minz, self.minz+max_diff)
        plt.show()


def main():
    graph = Graph()
    graph.add_node(Node())
    for i in range(1):
        graph.add_tunnel()
    graph.recalculate()
    points, normals = get_mesh_from_graph(graph)
    graph.plot()
    ptcl = open3d.geometry.PointCloud()
    ptcl.points = open3d.utility.Vector3dVector(points.T)
    ptcl.normals = open3d.utility.Vector3dVector(normals.T)
    poisson_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        ptcl, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    open3d.io.write_triangle_mesh("bpa_mesh.ply", poisson_mesh)
    open3d.visualization.draw_geometries([ptcl])


if __name__ == "__main__":
    main()
