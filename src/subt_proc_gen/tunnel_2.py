from subt_proc_gen.graph import Node, Edge, Graph
from subt_proc_gen.geometry import Point3D, Vector3D, Spline3D


class Tunnel:
    """A tunnel only contains nodes"""

    def __init__(self, nodes):
        if isinstance(nodes, list):
            nodes = tuple(nodes)
        for node in nodes:
            assert isinstance(node, Node)
        self._nodes = nodes
        self._spline = None
        self._hash = hash(frozenset(self._nodes))

    def __contains__(self, node: Node):
        return node in self._nodes

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Tunnel):
            return hash(self) == hash(other)
        return False

    def __iter__(self):
        return self._nodes

    def __getitem__(self, index):
        return self._nodes[index]

    def __len__(self):
        return len(self._nodes)

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([nd.xyz for nd in self._nodes])


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = set()
        self._tunnels_of_node = dict()

    def add_node(self, node: Node):
        super().add_node(node)
        if not node in self._tunnels_of_node:
            self._tunnels_of_node[node] = set()

    def remove_node(self, node: Node):
        super().remove_node(node)
        del self._tunnels_of_node[node]

    def add_tunnel(self, tunnel: Tunnel):
        self._tunnels.add(tunnel)
        for node in tunnel:
            self.add_node(node)
            self._tunnels_of_node[node].add(tunnel)
        for ni, nj in zip(tunnel[:-1], tunnel[1:]):
            self.connect(ni, nj)

    def remove_tunnel(self, tunnel: Tunnel):
        for node in tunnel:
            self._tunnels_of_node[node].remove(tunnel)
            if len(self._tunnels_of_node[node] == 0):
                self.remove_node(node)
