"""This file contains all the data structures that interpret the graph as a set of tunnels
and build it as such"""
from graph import Graph, Node, Edge
from generate_random_graph import Spline3D
from PARAMS import *
import numpy as np
class TunnelParams:
    def __init__(self, params=None):
        self.__dict__ = {"distance": 100,
                         "starting_direction": (1, 0, 0),
                         "horizontal_tendency": 0,
                         "horizontal_noise": 0,
                         "vertical_tendency": 0,
                         "vertical_noise": 0,
                         "min_seg_length": 10,
                         "max_seg_length": 15}
        if not params is None:
            assert isinstance(params, dict)
            for key in params.keys():
                self.__dict__[key] = params[key]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class Tunnel:
    def __init__(self, parent):
        assert isinstance(parent, Graph)
        self.parent = parent
        self.parent.tunnels.append(self)
        # The nodes should be ordered
        self.nodes = list()
        self._spline = None

    def split(self, node):
        assert node in self.nodes
        tunnel_1 = Tunnel(self.parent)
        tunnel_2 = Tunnel(self.parent)
        split_point = self.nodes.index(node)
        tunnel_1.set_nodes(self.nodes[:split_point+1])
        tunnel_2.set_nodes(self.nodes[split_point:])
        self.parent.remove_tunnel(self)

    def set_nodes(self, nodes):
        self.nodes = nodes
        self._spline = Spline3D([n.xyz for n in self.nodes])

    def add_node(self, node: Node):
        if len(self) != 0:
            self.nodes[-1].connect(node)
        node.add_tunnel(self)
        self.nodes.append(node)
        self.parent.add_node(node)
        self._spline = None

    @property
    def spline(self):
        if self._spline is None:
            self._spline = Spline3D([n.xyz for n in self.nodes])
        return self._spline

    def __len__(self):
        return len(self.nodes)

    @property
    def distance(self):
        return self.spline.distance

    def plot2d(self, ax):
        ds = np.arange(0, self.distance, SPLINE_PLOT_PRECISSION)
        xs, ys = [], []
        for d in ds:
            p, d = self.spline(d)
            x, y, z = p
            xs.append(x)
            ys.append(y)
        color = np.array(list(np.random.uniform(0.2, 0.75, size=3)))
        ax.plot(xs, ys, c=color, linewidth=3)


class Intersection:
    def __init__(self, parent, node):
        self.parent = parent
        self.node = node
        self.connected_tunnels = list()

    @property
    def n_tunnels(self):
        return len(self.connected_tunnels)