from subt_proc_gen.graph import Node, Edge, Graph
from subt_proc_gen.geometry import Point3D, Vector3D, Spline3D
from enum import Enum
import numpy as np
import math
import yaml


class ConnectorTunnelGenerationParams:
    _default_first_segment_vector = None
    _default_last_segment_vector = None
    _default_segment_length = 20
    _default_node_position_horizontal_noise = 7
    _default_node_position_vertical_noise = 5

    _random_segment_length_interval = (10, 20)
    _random_node_position_horizontal_noise_interval = (0, 15)
    _random_node_position_vertical_noise_interval = (0, 10)

    @classmethod
    def from_defaults(cls, first_segment_vector=None, last_segment_vector=None):
        return ConnectorTunnelGenerationParams(
            first_segment_vector=first_segment_vector,
            last_segment_vector=last_segment_vector,
            segment_length=cls._default_segment_length,
            node_position_horizontal_noise=cls._default_node_position_horizontal_noise,
            node_position_vertical_noise=cls._default_node_position_vertical_noise,
        )

    @classmethod
    def random(cls, first_segment_vector=None, last_segment_vector=None):
        min_segment_length = np.random.uniform(
            low=cls._random_segment_length_interval[0],
            high=cls._random_segment_length_interval[1],
        )
        node_position_horizontal_noise = np.random.uniform(
            low=cls._random_node_position_horizontal_noise_interval[0],
            high=cls._random_node_position_horizontal_noise_interval[1],
        )
        node_position_vertical_noise = np.random.uniform(
            low=cls._random_node_position_vertical_noise_interval[0],
            high=cls._random_node_position_vertical_noise_interval[1],
        )
        return ConnectorTunnelGenerationParams(
            first_segment_vector=first_segment_vector,
            last_segment_vector=last_segment_vector,
            segment_length=min_segment_length,
            node_position_horizontal_noise=node_position_horizontal_noise,
            node_position_vertical_noise=node_position_vertical_noise,
        )

    def __init__(
        self,
        first_segment_vector=None,
        last_segment_vector=None,
        segment_length=None,
        node_position_horizontal_noise=None,
        node_position_vertical_noise=None,
    ):
        assert not segment_length is None
        assert not node_position_horizontal_noise is None
        assert not node_position_vertical_noise is None
        self.first_segment_vector = first_segment_vector
        self.last_segment_vector = last_segment_vector
        self.segment_length = segment_length
        self.node_position_horizontal_noise = node_position_horizontal_noise
        self.node_position_vertical_noise = node_position_horizontal_noise

    def gen_random_displacement(self) -> Vector3D:
        x = np.random.uniform(
            -self.node_position_horizontal_noise, +self.node_position_horizontal_noise
        )
        y = np.random.uniform(
            -self.node_position_horizontal_noise, +self.node_position_horizontal_noise
        )
        z = np.random.uniform(
            -self.node_position_vertical_noise, +self.node_position_vertical_noise
        )
        return Vector3D((x, y, z))


class GrownTunnelGenerationParams:
    # Default params
    _default_distance = 100
    _default_initial_direction = Vector3D((1, 0, 0))
    _default_horizontal_tendency_rad = 0
    _default_vertical_tendency_rad = 0
    _default_horizontal_noise_rad = np.deg2rad(10)
    _default_vertical_noise_rad = np.deg2rad(5)
    _default_min_segment_length = 10
    _default_max_segment_lenght = 20
    # Random params
    _random_distance_range = (50, 300)
    _random_intial_inclination_range_deg = (-20, 20)
    _random_horizontal_tendency_range_deg = (-30, 30)
    _random_vertical_tendency_range_deg = (-20, 20)
    _random_horizontal_noise_range_deg = (0, 20)
    _random_vertical_noise_range_deg = (0, 10)

    @classmethod
    def from_defaults(cls, initial_direction=None):
        if initial_direction is None:
            initial_direction = cls._default_initial_direction
        return GrownTunnelGenerationParams(
            cls._default_distance,
            initial_direction,
            cls._default_horizontal_tendency_rad,
            cls._default_vertical_tendency_rad,
            cls._default_horizontal_noise_rad,
            cls._default_vertical_noise_rad,
            cls._default_min_segment_length,
            cls._default_max_segment_lenght,
        )

    @classmethod
    def random(cls, initial_direction=None):
        distance = np.random.uniform(
            cls._random_distance_range[0],
            cls._random_distance_range[1],
        )
        horizontal_tendency_rad = np.deg2rad(
            np.random.uniform(
                cls._random_horizontal_tendency_range_deg[0],
                cls._random_horizontal_tendency_range_deg[1],
            )
        )
        vertical_tendency_rad = np.deg2rad(
            np.random.uniform(
                cls._random_vertical_tendency_range_deg[0],
                cls._random_vertical_tendency_range_deg[1],
            )
        )
        horizontal_noise_rad = np.deg2rad(
            np.random.uniform(
                cls._random_horizontal_noise_range_deg[0],
                cls._random_horizontal_noise_range_deg[1],
            )
        )
        vertical_noise_rad = np.deg2rad(
            np.random.uniform(
                cls._random_vertical_noise_range_deg[0],
                cls._random_vertical_noise_range_deg[1],
            )
        )
        initial_inclination_rad = np.deg2rad(
            np.random.uniform(
                cls._random_intial_inclination_range_deg[0],
                cls._random_intial_inclination_range_deg[1],
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
        assert not distance is None
        assert not initial_direction is None
        assert not horizontal_tendency_rad is None
        assert not vertical_tendency_rad is None
        assert not horizontal_noise_rad is None
        assert not vertical_noise_rad is None
        assert not min_segment_length is None
        assert not max_segment_length is None
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
        return Tunnel(nodes, tunnel_type=TunnelType.grown)

    @classmethod
    def connector(
        cls, i_node: Node, f_node: Node, params: ConnectorTunnelGenerationParams
    ):
        if not params.first_segment_vector is None:
            nodes = [i_node, i_node + Vector3D(params.first_segment_vector)]
        else:
            nodes = [i_node]
        start_node = nodes[-1]
        if not params.last_segment_vector is None:
            final_nodes = [f_node + Vector3D(params.last_segment_vector), f_node]
        else:
            final_nodes = [f_node]
        finish_node = final_nodes[0]

        s_to_f_vector = finish_node - start_node
        n_segments = math.ceil(s_to_f_vector.length / params.segment_length)
        segment_length = s_to_f_vector.length / n_segments
        for n_segment in range(1, n_segments):
            nodes.append(
                start_node
                + Vector3D(s_to_f_vector.cartesian_unitary * n_segment * segment_length)
                + params.gen_random_displacement()
            )
        return Tunnel(
            nodes + final_nodes,
            tunnel_type=TunnelType.connector,
        )

    def __init__(
        self, nodes: list[Node] | tuple[Node], tunnel_type=TunnelType.from_nodes
    ):
        if isinstance(nodes, list):
            nodes = tuple(nodes)
        assert len(nodes) > 1
        for node in nodes:
            assert isinstance(node, Node)
        self._nodes = nodes
        self._tunnel_type = tunnel_type
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

    def set_tunnel_type(self, type):
        self._tunnel_type = TunnelType(type)

    @property
    def tunnel_type(self) -> TunnelType:
        return self._tunnel_type

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([nd._pose for nd in self._nodes])

    @property
    def nodes(self) -> list[Node]:
        return self._nodes


class NodeType(Enum):
    tunnel_node = 1
    multi_tunnel_intersection = 2
    self_intersection = 3
    self_loop_closing = 4
    end_of_tunnel = 5
    floating_node = 6
    tunnel_continuation_intersection = 7

    @property
    def inter_types(self):
        return [
            NodeType.multi_tunnel_intersection,
            NodeType.self_intersection,
            NodeType.self_loop_closing,
            NodeType.tunnel_continuation_intersection,
        ]


class TunnelNetwork(Graph):
    def __init__(self):
        super().__init__()
        self._tunnels = set()
        # Tracks the tunnels to which a node belongs
        self._tunnels_of_node = dict()
        # Tracks which tunnels are connected to each other
        self._node_types = dict()
        self._intersection_connectivity = dict()

    def add_node(self, node: Node):
        """IMPORTANT: This should only be called by _add_tunnel"""
        super().add_node(node)
        if not node in self._tunnels_of_node:
            self._tunnels_of_node[node] = set()

    def remove_node(self, node: Node):
        """IMPORTANT: This should only be called by _remove_tunnel"""
        super().remove_node(node)
        del self._tunnels_of_node[node]

    def _compute_node_type(self, node: Node):
        """A node is an intersection if either:
        1. It belongs to more than one tunnel
        2. If, being part of only one tunnel but:
            2.1 It connects with three nodes of said tunnel
            2.2 It is the first node of the tunnel and is connected with the last"""
        assert isinstance(node, Node)
        if len(self._tunnels_of_node[node]) > 1:
            if len(self.connected_nodes(node)) == 2:
                return NodeType.tunnel_continuation_intersection
            else:
                return NodeType.multi_tunnel_intersection
        elif len(self._tunnels_of_node[node]) == 1:
            if len(self.connected_nodes(node)) == 1:
                return NodeType.end_of_tunnel
            if len(self.connected_nodes(node)) > 2:
                return NodeType.self_intersection
            if len(self.connected_nodes(node)) == 2:
                position_in_tunnel = list(self._tunnels_of_node[node])[0].nodes.index(
                    node
                )
                if position_in_tunnel == 0:
                    return NodeType.self_loop_closing
                else:
                    return NodeType.tunnel_node
        else:
            return NodeType.floating_node

    def compute_node_types(self):
        for node in self.nodes:
            self._node_types[node] = self._compute_node_type(node)

    def compute_intersection_connections(self):
        for intersection in self.intersections:
            self._intersection_connectivity[intersection] = set()
            # Craw node by node along each of the connected nodes
            # until reaching next intersection node
            for node in self.connected_nodes(intersection):
                new_intersection_reached = False
                prev_node = intersection
                current_node = node
                while not new_intersection_reached:
                    if TODO:
                        pass
                    connected_to_current = list(self.connected_nodes(current_node))
                    connected_to_current.remove(prev_node)
                    nex_node = connected_to_current[0]

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
            if len(self._tunnels_of_node[node]) == 0:
                self.remove_node(node)

    def to_yaml(self, file):
        nodes_dict = {}
        for node in self.nodes:
            nodes_dict[node.id] = (float(node.x), float(node.y), float(node.z))
        tunnels_list = []
        for n_tunnel, tunnel in enumerate(self.tunnels):
            tunnels_list.append(
                {
                    "nodes": [node.id for node in tunnel.nodes],
                    "type": tunnel.tunnel_type.value,
                }
            )

        data = {"nodes": nodes_dict, "tunnels": tunnels_list}

        with open(file, "w+") as f:
            yaml.safe_dump(data, f)

    def load_yaml(self, file):
        with open(file, "r") as f:
            yaml_data = yaml.safe_load(f)
        nodes = {}
        for node_id in yaml_data["nodes"]:
            nodes[node_id] = Node(yaml_data["nodes"][node_id])
        for tunnel in yaml_data["tunnels"]:
            self.add_tunnel(
                Tunnel(
                    [nodes[node_id] for node_id in tunnel["nodes"]],
                    tunnel_type=TunnelType(tunnel["type"]),
                )
            )

    @property
    def tunnels(self) -> set[Tunnel]:
        return self._tunnels

    @property
    def intersections(self) -> set[Node]:
        self.compute_node_types()
        intersections = set()
        for node in self._nodes:
            if self._node_types(node) in NodeType.inter_types:
                intersections.add(node)
        return intersections
