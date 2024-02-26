from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, Node
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from random import randrange


for i in range(20):
    while True:
        try:
            tn = TunnelNetwork(initial_node=False)
            n1 = Node(0, 0, randrange(-5, 5))
            while True:
                t1 = Tunnel.grown(i_node=n1)
                if not tn.check_collisions(t1):
                    tn.add_tunnel(t1)
                    break
            while True:
                t2 = Tunnel.grown(i_node=tn.get_random_node())
                if not tn.check_collisions(t2):
                    tn.add_tunnel(t2)
                    break
            while True:
                ni = tn.get_random_node()
                nf = tn.get_random_node()
                t3 = Tunnel.connector(inode=ni, fnode=nf)
                if not tn.check_collisions(t3):
                    tn.add_tunnel(t3)
                    break
            tnmg = TunnelNetworkMeshGenerator(tn)
            tnmg.compute_all()
            break
        except:
            pass

    import pyvista as pv
    from pyvista.plotting.plotting import Plotter
    import numpy as np
    from subt_proc_gen.display_functions import plot_spline, plot_nodes

    plotter = Plotter(off_screen=True)
    # plotter = Plotter()
    plotter.set_background("w")
    plotter.add_mesh(tnmg.pyvista_mesh, style="wireframe")
    plot_spline(plotter, t1.spline, color="purple", radius=0.5)
    plot_spline(plotter, t2.spline, color="orange", radius=0.5)
    plot_spline(plotter, t3.spline, color="yellow", radius=0.5)
    plot_nodes(plotter, tn.nodes, radius=1, color="k")
    plotter.add_lines(np.array(((0, 0, 0), (100, 0, 0))), color="r")
    plotter.add_lines(np.array(((0, 0, 0), (0, 100, 0))), color="g")
    plotter.add_lines(np.array(((0, 0, 0), (0, 0, 100))), color="b")
    plotter.camera_position = [
        (-215.4763287253459, -118.79306719122829, 166.90769689228273),
        (0.8052306955758812, 23.495406259788453, 2.3282761218368897),
        (0.4592542245702642, 0.27783968494893313, 0.843736135697555),
    ]
    import os

    images_path = "/home/lorenzo/images/papers/subt_proc_gen"
    os.makedirs(images_path, exist_ok=True)
    plotter.show(screenshot=os.path.join(images_path, f"tn_procedural_{i}.png"))
    # plotter.show()
    # print(plotter.camera_position)
