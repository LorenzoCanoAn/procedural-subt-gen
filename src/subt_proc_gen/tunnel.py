from subt_proc_gen.graph import Node, Graph
from subt_proc_gen.geometry import Vector3D, Spline3D, check_spline_collision
from enum import Enum
import numpy as np
import math
import yaml


class ConnectorTunnelGenerationParams:
    _default_segment_length = 20
    _default_node_position_horizontal_noise = 7
    _default_node_position_vertical_noise = 5

    _random_segment_length_interval = (10, 20)
    _random_node_position_horizontal_noise_interval = (0, 15)
    _random_node_position_vertical_noise_interval = (0, 10)

    @classmethod
    def from_defaults(cls):
        return ConnectorTunnelGenerationParams(
            segment_length=cls._default_segment_length,
            node_position_horizontal_noise=cls._default_node_position_horizontal_noise,
            node_position_vertical_noise=cls._default_node_position_vertical_noise,
        )

    @classmethod
    def random(cls):
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
            segment_length=min_segment_length,
            node_position_horizontal_noise=node_position_horizontal_noise,
            node_position_vertical_noise=node_position_vertical_noise,
        )

    def __init__(
        self,
        segment_length=None,
        node_position_horizontal_noise=None,
        node_position_vertical_noise=None,
    ):
        assert not segment_length is None
        assert not node_position_horizontal_noise is None
        assert not node_position_vertical_noise is None
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
    _default_horizontal_tendency_rad = 0
    _default_vertical_tendency_rad = 0
    _default_horizontal_noise_rad = np.deg2rad(10)
    _default_vertical_noise_rad = np.deg2rad(5)
    _default_min_segment_length = 10
    _default_max_segment_lenght = 20
    # Random params
    _random_distance_range = (50, 300)
    _random_vertical_tendency_range_deg = (-20, 20)
    _random_horizontal_tendency_range_deg = (-40, 40)
    _random_horizontal_noise_range_deg = (0, 20)
    _random_vertical_noise_range_deg = (0, 10)
    _random_min_segment_length_fraction_range = (0.05, 0.15)
    _random_max_segment_length_fraction_range = (0.15, 0.30)

    @classmethod
    def from_defaults(cls):
        return GrownTunnelGenerationParams(
            cls._default_distance,
            cls._default_horizontal_tendency_rad,
            cls._default_vertical_tendency_rad,
            cls._default_horizontal_noise_rad,
            cls._default_vertical_noise_rad,
            cls._default_min_segment_length,
            cls._default_max_segment_lenght,
        )

    @classmethod
    def random(cls):
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
        min_seg_length_frac = np.random.uniform(
            cls._random_min_segment_length_fraction_range[0],
            cls._random_min_segment_length_fraction_range[1],
        )
        max_seg_length_frac = np.random.uniform(
            cls._random_max_segment_length_fraction_range[0],
            cls._random_max_segment_length_fraction_range[1],
        )
        return GrownTunnelGenerationParams(
            distance=distance,
            horizontal_tendency_rad=horizontal_tendency_rad,
            vertical_tendency_rad=vertical_tendency_rad,
            horizontal_noise_rad=horizontal_noise_rad,
            vertical_noise_rad=vertical_noise_rad,
            min_segment_length=distance * min_seg_length_frac,
            max_segment_length=distance * max_seg_length_frac,
        )

    def __init__(
        self,
        distance=None,
        horizontal_tendency_rad=None,
        vertical_tendency_rad=None,
        horizontal_noise_rad=None,
        vertical_noise_rad=None,
        min_segment_length=None,
        max_segment_length=None,
    ):
        assert not distance is None
        assert not horizontal_tendency_rad is None
        assert not vertical_tendency_rad is None
        assert not horizontal_noise_rad is None
        assert not vertical_noise_rad is None
        assert not min_segment_length is None
        assert not max_segment_length is None
        self.distance = distance
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
    def grown(
        cls, i_node: Node, i_direction: Vector3D, params: GrownTunnelGenerationParams
    ):
        nodes = [i_node]
        prev_direction = Vector3D(i_direction)
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
        cls,
        i_node: Node,
        f_node: Node,
        params: ConnectorTunnelGenerationParams,
        i_vector: Vector3D = None,
        f_vector: Vector3D = None,
    ):
        if not i_vector is None:
            nodes = [i_node, i_node + Vector3D(i_vector)]
        else:
            nodes = [i_node]
        start_node = nodes[-1]
        if not f_vector is None:
            final_nodes = [f_node + Vector3D(f_vector), f_node]
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
        return self._spline

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

    @classmethod
    def inter_types(self):
        return [
            NodeType.multi_tunnel_intersection,
            NodeType.self_intersection,
            NodeType.self_loop_closing,
            NodeType.tunnel_continuation_intersection,
        ]


class TunnelNetworkParams:
    _default_collision_distance = 10
    _default_min_intersection_angle = np.deg2rad(30)
    _default_max_inclination = np.deg2rad(30)

    def __init__(
        self, collision_distance=None, max_inclination=None, min_intersection_angle=None
    ):
        assert not collision_distance is None
        assert not max_inclination is None
        assert not min_intersection_angle is None
        self.collision_distance = collision_distance
        self.min_intersection_angle = min_intersection_angle
        self.max_inclination = max_inclination

    @classmethod
    def from_defaults(cls):
        return cls(
            collision_distance=cls._default_collision_distance,
            max_inclination=cls._default_max_inclination,
            min_intersection_angle=cls._default_min_intersection_angle,
        )


class TunnelNetwork(Graph):
    def __init__(self, params=None, initial_node=True):
        super().__init__()
        if params is None:
            self._params = TunnelNetworkParams.from_defaults()
        self._tunnels = set()
        # Tracks the tunnels to which a node belongs
        self._tunnels_of_node = dict()
        # Tracks which tunnels are connected to each other
        self._node_types = dict()
        if initial_node:
            self.add_node(Node((0, 0, 0)))

    @property
    def params(self) -> TunnelNetworkParams:
        return self._params

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
            elif len(self.connected_nodes(node)) > 2:
                return NodeType.multi_tunnel_intersection
            else:
                raise Exception(
                    f"Node {node} has multiple tunnels but less than one connection"
                )
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

    def compute_intersection_connectivity_graph(self) -> Graph:
        intersection_connectivity_graph = Graph()
        for intersection in self.intersections:
            intersection_connectivity_graph.add_node(intersection)
            # Craw node by node along each of the connected nodes
            # until reaching next intersection node
            for node in self.connected_nodes(intersection):
                current_node = intersection
                next_node = node
                new_intersection_reached = False
                while not new_intersection_reached:
                    if (
                        self._node_types[next_node] in NodeType.inter_types()
                        or self._node_types[next_node] == NodeType.end_of_tunnel
                    ):
                        intersection_connectivity_graph.add_node(next_node)
                        intersection_connectivity_graph.connect(intersection, next_node)
                        new_intersection_reached = True
                    else:
                        nodes_connected_to_next_node = list(
                            self.connected_nodes(next_node)
                        )
                        nodes_connected_to_next_node.remove(current_node)
                        assert len(nodes_connected_to_next_node) == 1
                        current_node = next_node
                        next_node = nodes_connected_to_next_node[0]
        return intersection_connectivity_graph

    def add_tunnel(
        self,
        tunnel: Tunnel,
    ):
        self._tunnels.add(tunnel)
        for node in tunnel:
            self.add_node(node)
            self._tunnels_of_node[node].add(tunnel)
        for ni, nj in zip(tunnel[:-1], tunnel[1:]):
            self.connect(ni, nj)

    def check_collisions(
        self,
        tunnel: Tunnel,
        collision_distance,
    ):
        for tunnel_i in self.tunnels:
            if tunnel_i is tunnel:
                continue
            common_nodes = []
            for node in tunnel.nodes:
                if node in tunnel_i.nodes:
                    common_nodes.append(node)
            if check_spline_collision(
                tunnel.spline,
                tunnel_i.spline,
                collision_distance,
                omision_points=[node.xyz for node in common_nodes],
                omision_distances=[collision_distance * 2 for _ in common_nodes],
            ):
                return True
        return False

    def remove_tunnel(self, tunnel: Tunnel):
        for node in tunnel:
            self._tunnels_of_node[node].remove(tunnel)
            if len(self._tunnels_of_node[node]) == 0:
                self.remove_node(node)
        self._tunnels.remove(tunnel)

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
        Node.set_global_counter(max(yaml_data["nodes"]))

    def add_random_grown_tunnel(
        self,
        params=None,
        n_trials=10,
        collsion_distance=None,
        max_inclination=None,
        min_intersection_angle=None,
    ):
        if params is None:
            params = GrownTunnelGenerationParams.random()
        n = 0
        successful = False
        if collsion_distance is None:
            collision_distance = self.params.collision_distance
        if max_inclination is None:
            max_inclination = self.params.max_inclination
        if min_intersection_angle is None:
            min_intersection_angle = self.params.min_intersection_angle
        while not successful and n < n_trials:
            n += 1
            i_node = np.random.choice(np.array(list(self.nodes)))
            tunnel = Tunnel.grown(
                i_node=i_node,
                i_direction=Vector3D.random(
                    inclination_range=(np.deg2rad(-10), np.deg2rad(10)),
                    yaw_range=(0, np.pi * 2),
                ),
                params=params,
            )
            self.add_tunnel(tunnel)
            collides = self.check_collisions(
                tunnel, collision_distance=collision_distance
            )
            too_much_inclination = check_maximum_inclination_of_spline(
                tunnel.spline, max_inclination
            )
            too_small_angle = False
            for tunnel_j in self._tunnels_of_node[i_node]:
                if not tunnel_j is tunnel:
                    angle = min_angle_between_splines_at_point(
                        tunnel.spline, tunnel_j.spline, i_node._pose.xyz
                    )
                    too_small_angle = angle < min_intersection_angle
                    if too_small_angle:
                        break
            if collides or too_much_inclination or too_small_angle:
                successful = False
                self.remove_tunnel(tunnel)
            else:
                successful = True

    @property
    def tunnels(self) -> set[Tunnel]:
        return self._tunnels

    @property
    def intersections(self) -> set[Node]:
        self.compute_node_types()
        intersections = set()
        for node in self._nodes:
            if self._node_types[node] in NodeType.inter_types():
                intersections.add(node)
        return intersections


def check_maximum_inclination_of_spline(spline: Spline3D, max_inc_rad, precision=0.5):
    vs = spline.discretize(precision)[2]
    inc = np.arctan2(vs[:, 2], np.linalg.norm(vs[:, :2], axis=1))
    return np.any(inc > max_inc_rad)


def min_angle_between_splines_at_point(
    spline1: Spline3D, spline2: Spline3D, point, precision=0.1
):
    _, ap1, av1 = spline1.discretize(precision)
    _, ap2, av2 = spline2.discretize(precision)
    id1 = np.argmin(np.linalg.norm(ap1 - point, axis=1))
    id2 = np.argmin(np.linalg.norm(ap2 - point, axis=1))
    v1xy = np.reshape(av1[id1, 0:2], (1, 2))
    v2xy = np.reshape(av2[id2, 0:2], (1, 2))
    sign = [-1, 1]
    angle_between = np.pi * 2
    for dir1 in sign:
        for dir2 in sign:
            v1dir = dir1 * v1xy
            v2dir = dir2 * v2xy
            angle_between = min(angle_between, np.arccos(np.dot(v1dir, v2dir.T)))
    return angle_between
