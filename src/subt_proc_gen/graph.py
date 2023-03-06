"""This file contains the graph structure onto which evertything else is built upon"""
import numpy as np


class Edge:
    def __init__(self, n0, n1):
        assert issubclass(type(n0), Node)
        assert issubclass(type(n1), Node)

        self.nodes = frozenset((n0, n1))

    def __getitem__(self, index):
        return list(self.nodes[index])

    def __hash__(self):
        return hash(self.nodes)

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
    _graph = None

    def __init__(self, coords=np.zeros(3)):
        self._graph.add_node(self)
        self._connected_nodes = set()
        self.coords = coords

    @classmethod
    def set_graph(cls, graph):
        cls.__graph = graph

    def connect(self, node):
        """This function does everything to connect two nodes and update the
        info in all relevant places"""
        assert isinstance(node, Node)
        self.add_connection(node)
        node.add_connection(self)

    def add_connection(self, node):
        """This function only inserts a new node in the connected nodes set"""
        self._connected_nodes.add(node)

    def remove_connection(self, node):
        """This function only deletes a node from the connected nodes set"""
        self._connected_nodes.remove(node)

    def delete(self):
        for node in self._connected_nodes:
            assert isinstance(node, Node)
            node.remove_connection(self)
        self._graph._nodes.remove(self)

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

    @property
    def connected_nodes(self):
        return self._connected_nodes


class Graph:
    def __init__(self):
        Node._graph = self
        self.recalculate_edges = True
        self._nodes = set()
        self._edges = set()

    def add_node(self, node):
        if not node in self._nodes:
            self._nodes.add(node)

    def remove_node(self, node: Node):
        assert node in self._nodes, Exception(
            "Trying to remove a node that is not part of the graph"
        )
        node.delete()
        self._nodes.remove(node)
        self.recalculate_edges = True

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def edges(self):
        if self.recalculate_edges:
            self._edges.clear()
            for node_i in self._nodes:
                assert isinstance(node_i, Node)
                for connected_node in node_i._connected_nodes:
                    self._edges.add(Edge(node_i, connected_node))

        return self._edges

    @property
    def minx(self):
        return min([n.x for n in self._nodes])

    @property
    def miny(self):
        return min([n.y for n in self._nodes])

    @property
    def minz(self):
        return min([n.z for n in self._nodes])

    @property
    def maxx(self):
        return max([n.x for n in self._nodes])

    @property
    def maxy(self):
        return max([n.y for n in self._nodes])

    @property
    def maxz(self):
        return max([n.z for n in self._nodes])
