from subt_proc_gen.tunnel import TunnelNetwork, Tunnel
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh

tn = TunnelNetwork()
tn.add_tunnel(Tunnel(initial_pose=(0, 0, 0)))
while tn.num_of_tunnels() < 3:
    (node,) = tn.get_random_node()
    tn.add_tunnel(Tunnel(inode=node))
while tn.num_of_tunnels() < 4:
    n1, n2 = tn.get_random_node(num_of_nodes=2)
    tn.add_tunnel(Tunnel(inode=n1, fnode=n2))
tnwm = TunnelNetworkWithMesh(tn, roughness=0.01)
tnwm.clean_intersections()
tnwm.compute_mesh()
tnwm.save_mesh("mesh.obj")
