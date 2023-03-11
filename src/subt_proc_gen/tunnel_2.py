from subt_proc_gen.graph import Node, Edge, Graph
from subt_proc_gen.geometry import Point3D, Vector3D, Spline3D


class ConnectorTunnelGenerationParams:
    def __init__(
        self,
        first_segment_direction=None,
        last_segment_direction=None,
        min_segment_length=None,
        node_position_horizontal_noise=None,
        node_position_vertical_noise=None,
    ):
        assert (
            not min_segment_length is None
            and not node_position_horizontal_noise is None
            and not node_position_vertical_noise is None
        )
        self.first_segment_direction = first_segment_direction
        self.last_segment_direction = last_segment_direction
        self.min_segment_length = min_segment_length
        self.node_position_horizontal_noise = node_position_horizontal_noise
        self.node_position_vertical_noise = node_position_horizontal_noise


class GrownTunnelGenerationParams:
    def __init__(
        self,
        distance=None,
        starting_direction=None,
        horizontal_tendency=None,
        vertical_tendency=None,
        horizontal_noise=None,
        vertical_noise=None,
        segment_length=None,
        segment_length_noise=None,
    ):
        assert (
            not distance is None
            and not starting_direction is None
            and not horizontal_tendency is None
            and not vertical_tendency is None
            and not horizontal_noise is None
            and not vertical_noise is None
            and not segment_length is None
            and not segment_length_noise is None
        )
        self.distance = distance
        self.starting_direction = starting_direction
        self.horizontal_tendency = horizontal_tendency
        self.vertical_tendency = vertical_tendency
        self.horizontal_noise = horizontal_noise
        self.vertical_noise = vertical_noise
        self.segment_length = segment_length
        self.segment_length_noise = segment_length_noise


class Tunnel:
    """A tunnel only contains nodes"""

    @classmethod
    def grown(cls, node: Node, params: GrownTunnelGenerationParams):
        pass

    @classmethod
    def connector(
        cls, i_node: Node, f_node: Node, params: ConnectorTunnelGenerationParams
    ):
        pass

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

    def _add_tunnel(self, tunnel: Tunnel):
        self._tunnels.add(tunnel)
        for node in tunnel:
            self.add_node(node)
            self._tunnels_of_node[node].add(tunnel)
        for ni, nj in zip(tunnel[:-1], tunnel[1:]):
            self.connect(ni, nj)

    def _remove_tunnel(self, tunnel: Tunnel):
        for node in tunnel:
            self._tunnels_of_node[node].remove(tunnel)
            if len(self._tunnels_of_node[node] == 0):
                self.remove_node(node)
