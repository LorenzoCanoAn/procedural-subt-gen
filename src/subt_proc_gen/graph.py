"""This file contains the graph structure onto which evertything else is built upon"""
import numpy as np


class Node:
    def __init__(self, coords=None):
        if coords is None:
            self._coords = np.zeros((1, 3), dtype=np.double)
        else:
            self.set_coords(coords)

    def set_coords(self, coords):
        if isinstance(coords, list) or isinstance(coords, tuple):
            assert len(coords) == 3
            self._coords = np.reshape(np.array(coords, dtype=np.ndarray), (1, 3))
        elif isinstance(coords, np.ndarray):
            assert coords.size == 3
            self._coords = np.reshape(coords.astype(np.double), (1, 3))

    @property
    def xyz(self):
        return self._coords

    @property
    def x(self):
        return self._coords[0]

    @property
    def y(self):
        return self._coords[1]

    @property
    def z(self):
        return self._coords[2]


class Edge:
    def __init__(self, n0: Node, n1: Node):
        self._nodes = frozenset((n0, n1))

    def __hash__(self):
        return hash(self._nodes)

    @property
    def nodes(self):
        return self._nodes


class Graph:
    def __init__(self):
        self._nodes = set()
        self._edges = set()
        self._adj_list = dict()
        self._recalculate_edges = True

    def add_node(self, node: Node):
        assert isinstance(node, Node)
        if not node in self._nodes:
            self._nodes.add(node)
            self._adj_list[node] = set()
        self.recalculate_edges = True

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def connect(self, ni: Node, nj: Node):
        if ni is nj:
            return
        self._adj_list[ni].add(nj)
        self._adj_list[nj].add(ni)
        self.recalculate_edges = True

    def remove_node(self, node: Node):
        assert node in self._nodes
        for cn in self._adj_list[node]:
            self._adj_list[cn].remove(node)
        del self._adj_list[node]
        self._nodes.remove(node)
        self.recalculate_edges = True

    def disconnect(self, ni: Node, nj: Node):
        if ni is nj:
            return
        self._adj_list[ni].remove(nj)
        self._adj_list[nj].remove(ni)
        self.recalculate_edges = True

    def are_connected(self, node1: Node, node2: Node):
        return node2 in self._adj_list[node1]

    def assert_unidirectionality(self):
        for ni in self._adj_list:
            for nj in self._adj_list[ni]:
                assert ni in self._adj_list[nj]

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        if self.recalculate_edges:
            self._edges = set()
            for ni in self._adj_list:
                for nj in self._adj_list[ni]:
                    self._edges.add(Edge(ni, nj))
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


def test1():
    g = Graph()
    n1 = Node()
    n2 = Node(np.random.random(3))
    n3 = Node([1, 2, 3])
    n4 = Node((1, 2, 3))


def test2():
    g = Graph()
    for _ in range(1000):
        g.add_node(Node(np.random.random(3)))
    for _ in range(500):
        ni, nj = random.choices(tuple(g.nodes), k=2)
        g.connect(ni, nj)

    edges = tuple(g.edges)
    to_remove = random.choices(edges, k=100)

    for n_edge, edge in enumerate(set(to_remove)):
        print(n_edge)
        assert isinstance(edge, Edge)
        n1, n2 = tuple(edge.nodes)
        g.disconnect(n1, n2)
    g.assert_unidirectionality()


# TESTS
if __name__ == "__main__":
    import random

    test1()
    test2()
