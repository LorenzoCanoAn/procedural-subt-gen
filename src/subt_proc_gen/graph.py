"""This file contains the graph structure onto which evertything else is built upon"""
import numpy as np
from subt_proc_gen.geometry import Point


class Node:
    _global_counter = 0

    @classmethod
    def _get_next_id(cls):
        id = cls._global_counter
        cls._global_counter += 1
        return id

    def __init__(self, coords=None):
        self._id = self._get_next_id()
        if coords is None:
            self._pose = Point()
        else:
            self._pose = Point(coords)

    def __str__(self):
        return str(self._id)

    @property
    def xyz(self):
        return self._pose.xyz

    @property
    def x(self):
        return self._pose.x

    @property
    def y(self):
        return self._pose.y

    @property
    def z(self):
        return self._pose.z


class Edge:
    def __init__(self, n0: Node, n1: Node):
        self._nodes = frozenset((n0, n1))
        self._hash = hash(self._nodes)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        n1, n2 = self.nodes
        return f"({n1}, {n2})"

    @property
    def nodes(self):
        return tuple(self._nodes)


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


###########################################################################
# FUNCTIONS
###########################################################################


def assert_unidirectionality(graph: Graph):
    for ni in graph._adj_list:
        for nj in graph._adj_list[ni]:
            assert ni in graph._adj_list[nj]


###########################################################################
# TESTS
###########################################################################


def test1():
    g = Graph()
    n1 = Node()
    n2 = Node(np.random.random(3))
    n3 = Node([1, 2, 3])
    n4 = Node((1, 2, 3))


def test2():
    g = Graph()
    for _ in range(20000):
        g.add_node(Node(np.random.random(3)))
    for _ in range(1000):
        ni, nj = random.choices(tuple(g.nodes), k=2)
        g.connect(ni, nj)
    edges = tuple(g.edges)
    to_remove = set(list(set(random.choices(edges, k=500))))
    for n_edge, edge in enumerate(set(to_remove)):
        assert isinstance(edge, Edge)
        n1, n2 = tuple(edge.nodes)
        g.disconnect(n1, n2)
    assert_unidirectionality(g)


# TESTS
if __name__ == "__main__":
    import random

    test1()
    test2()
