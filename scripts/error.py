import numpy as np
import pyvista as pv
from src.subt_proc_gen.tunnel import TunnelNetwork
from pointcloud_from_graph import pc_from_graph
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import *
from subt_proc_gen.display_functions import debug_plot
import matplotlib.pyplot as plt

graph = TunnelNetwork()

a = CaveNode(np.array([0, 0, 0]))
b = CaveNode(np.array([50, 0, 0]))
c = CaveNode(np.array([100, 0, 0]))
d = CaveNode(np.array([100, 100, 0]))
e = CaveNode(np.array([50, 100, 0]))
f = CaveNode(np.array([0, 100, 0]))
h = CaveNode(np.array([50, 50, 0]))
nodes = [a, b, c, d, e, f]

t = Tunnel(graph)
t.set_nodes(nodes)

nodes2 = [b, h, e]
t1 = Tunnel(graph)
t1.set_nodes(nodes2)
plotter = pv.Plotter()

pc_from_graph(plotter, 0.01, graph, "./mesh.obj", radius=2)
plotter.show()
