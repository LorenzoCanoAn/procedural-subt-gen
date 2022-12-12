"""This file contains the graph structure onto which evertything else is built upon"""
import numpy as np


class Edge:
    def __init__(self, n0, n1):
        self.nodes = {n0, n1}
        self.subnodes = set()
        n0.add_connection(n1)
        n1.add_connection(n0)

    def __getitem__(self, index):
        return self.nodes[index]

    def plot2d(self, ax):
        nodes = list(self.nodes)
        x0 = nodes[0].x
        x1 = nodes[1].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        ax.plot([x0, x1], [y0, y1], c="k")

    def plot3d(self, ax):
        nodes = list(self.nodes)
        x0 = nodes[0].x
        x1 = nodes[1].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        z0 = nodes[0].z
        z1 = nodes[1].z
        ax.plot3D([x0, x1], [y0, y1], [z0, z1], c="k")

class Node:
    __graph = None

    def __init__(self, coords=np.zeros(3), graph=None):
        if graph is None:
            assert isinstance(self.__graph, Graph)
            self.graph = self.__graph
        else:
            self.graph = graph
        self.connected_nodes = set()
        self.coords = coords
        self.tunnels = set()

    @classmethod
    def set_graph(cls, graph):
        cls.__graph = graph

    def add_tunnel(self, new_tunnel):
        """Nodes must keep track of what tunnels they are a part of
        this way, if a node is part of more than one tunnel, it means it
        is  an intersection"""
        if len(self.tunnels) == 0:
            self.tunnels.add(new_tunnel)
        if len(self.tunnels) == 1:
            if len(self.connected_nodes) == 2:
                list(self.tunnels)[0].split(self)

    def connect(self, node):
        """This function does everything to connect two nodes and update the 
        info in all relevant places"""
        assert isinstance(node, Node)
        self.graph.edges.append(Edge(self, node))

    def add_connection(self, node):
        """This function only inserts a new node in the connected nodes set"""
        self.connected_nodes.add(node)

    def remove_connection(self, node):
        """This function only deletes a node from the connected nodes set"""
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

class Graph:
    def __init__(self):
        self.recalculate_control = True
        self.nodes = list()
        self.edges = list()

        self.tunnels = []  
        self.intersections = []

    def add_node(self, node):
        self.recalculate_control = True
        if not node in self.nodes:
            self.nodes.append(node)

    def remove_tunnel(self, tunnel):
        assert tunnel in self.tunnels
        self.tunnels.remove(tunnel)


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

    