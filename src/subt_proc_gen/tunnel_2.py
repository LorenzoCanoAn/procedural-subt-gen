from subt_proc_gen.graph import Node, Edge, Graph
from subt_proc_gen.geometry import Point3D, Vector3D, Spline3D
from enum import Enum
import numpy as np


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
    default_distance = 100
    default_initial_direction = Vector3D((1, 0, 0))
    default_horizontal_tendency_rad = 0
    default_vertical_tendency_rad = 0
    default_horizontal_noise_rad = np.deg2rad(10)
    default_vertical_noise_rad = np.deg2rad(5)
    default_min_segment_length = 10
    default_max_segment_lenght = 20

    random_distance_range = (50, 300)
    random_intial_inclination_range_deg = (-20, 20)
    random_horizontal_tendency_range_deg = (-30, 30)
    random_vertical_tendency_range_deg = (-20, 20)
    random_horizontal_noise_range_deg = (0, 20)
    random_vertical_noise_range_deg = (0, 10)

    @classmethod
    def from_defaults(cls, initial_direction=None):
        if initial_direction is None:
            initial_direction = cls.default_initial_direction
        return GrownTunnelGenerationParams(
            cls.default_distance,
            cls.default_initial_direction,
            cls.default_horizontal_tendency_rad,
            cls.default_vertical_tendency_rad,
            cls.default_horizontal_noise_rad,
            cls.default_vertical_noise_rad,
            cls.default_min_segment_length,
            cls.default_max_segment_lenght,
        )

    @classmethod
    def random(cls, initial_direction=None):
        distance = np.random.uniform(
            cls.random_distance_range[0],
            cls.random_distance_range[1],
        )

        horizontal_tendency_rad = np.deg2rad(
            np.random.uniform(
                cls.random_horizontal_tendency_range_deg[0],
                cls.random_horizontal_tendency_range_deg[1],
            )
        )

        vertical_tendency_rad = np.deg2rad(
            np.random.uniform(
                cls.random_vertical_tendency_range_deg[0],
                cls.random_vertical_tendency_range_deg[1],
            )
        )

        horizontal_noise_rad = np.deg2rad(
            np.random.uniform(
                cls.random_horizontal_noise_range_deg[0],
                cls.random_horizontal_noise_range_deg[1],
            )
        )
        vertical_noise_rad = np.deg2rad(
            np.random.uniform(
                cls.random_vertical_noise_range_deg[0],
                cls.random_vertical_noise_range_deg[1],
            )
        )
        initial_inclination_rad = np.deg2rad(
            np.random.uniform(
                cls.random_intial_inclination_range_deg[0],
                cls.random_intial_inclination_range_deg[1],
            )
        )
        initial_yaw = np.random.uniform(0, np.pi * 2)
        return GrownTunnelGenerationParams(
            distance=distance,
            initial_direction=Vector3D.from_inclination_yaw_length(
                initial_inclination_rad, initial_yaw, 1
            ),
            horizontal_tendency_rad=horizontal_tendency_rad,
            vertical_tendency_rad=vertical_tendency_rad,
            horizontal_noise_rad=horizontal_noise_rad,
            vertical_noise_rad=vertical_noise_rad,
            min_segment_length=distance * 0.1,
            max_segment_length=distance * 0.4,
        )

    def __init__(
        self,
        distance=None,
        initial_direction=None,
        horizontal_tendency_rad=None,
        vertical_tendency_rad=None,
        horizontal_noise_rad=None,
        vertical_noise_rad=None,
        min_segment_length=None,
        max_segment_length=None,
    ):
        assert (
            not distance is None
            and not initial_direction is None
            and not horizontal_tendency_rad is None
            and not vertical_tendency_rad is None
            and not horizontal_noise_rad is None
            and not vertical_noise_rad is None
            and not min_segment_length is None
            and not max_segment_length is None
        )
        self.distance = distance
        self.starting_direction = initial_direction
        self.horizontal_tendency = horizontal_tendency_rad
        self.vertical_tendency = vertical_tendency_rad
        self.horizontal_noise = horizontal_noise_rad
        self.vertical_noise = vertical_noise_rad
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length

    def __str__(self):
        return f"""distance: {self.distance}
starting_direction: {self.starting_direction}
horizontal_tendency: {self.horizontal_tendency}
vertical_tendency: {self.vertical_tendency}
horizontal_noise: {self.horizontal_noise}
vertical_noise: {self.vertical_noise}
min_segment_length: {self.min_segment_length}
max_segment_length: {self.max_segment_length}"""

    def get_segment_length(self):
        return np.random.uniform(self.min_segment_length, self.max_segment_length)

    def get_inclination_angle(self):
        return np.random.uniform(
            self.vertical_tendency - self.vertical_noise,
            self.vertical_tendency + self.vertical_noise,
        )

    def get_horizontal_angle(self):
        return np.random.uniform(
            self.horizontal_tendency - self.horizontal_noise,
            self.horizontal_tendency + self.horizontal_noise,
        )

    def get_new_direction(self, prev_dir: Vector3D) -> Vector3D:
        prev_yaw = prev_dir.theta
        new_yaw = prev_yaw + self.get_horizontal_angle()
        new_inclination = self.get_inclination_angle()
        new_length = self.get_segment_length()
        return Vector3D.from_inclination_yaw_length(
            new_inclination, new_yaw, new_length
        )


class TunnelType(Enum):
    grown = 1
    connector = 2
    from_nodes = 3


class Tunnel:
    """A tunnel only contains nodes"""

    @classmethod
    def grown(cls, i_node: Node, params: GrownTunnelGenerationParams):
        nodes = [i_node]
        prev_direction = Vector3D(params.starting_direction)
        prev_direction.set_distance(params.get_segment_length())
        d = prev_direction.length
        nodes.append(nodes[-1] + prev_direction)
        while d < params.distance:
            new_dir = params.get_new_direction(prev_direction)
            nodes.append(nodes[-1] + new_dir)
            d += new_dir.length
        return Tunnel(nodes)

    @classmethod
    def connector(
        cls, i_node: Node, f_node: Node, params: ConnectorTunnelGenerationParams
    ):
        pass

    def __init__(self, nodes: list[Node]):
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
        return self._nodes.__iter__()

    def __getitem__(self, index):
        return self._nodes[index]

    def __len__(self):
        return len(self._nodes)

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([nd._pose for nd in self._nodes])


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = set()
        # Tracks the tunnels to which a node belongs
        self._tunnels_of_node = dict()
        # Tracks which tunnels are connected to each other
        self._tunnel_connections = dict()

    def add_node(self, node: Node):
        """IMPORTANT: This should only be called by _add_tunnel"""
        super().add_node(node)
        if not node in self._tunnels_of_node:
            self._tunnels_of_node[node] = set()

    def remove_node(self, node: Node):
        """IMPORTANT: This should only be called by _remove_tunnel"""
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
