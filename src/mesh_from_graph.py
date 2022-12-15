import pickle
from mesh_generation import *
import open3d as o3d
from PARAMS import *
import os
import shutil
from tunnel import Tunnel, TunnelNetwork
MESH_FOLDER = "meshes"
def tunnel_interesects_with_list(tunnel:Tunnel, list_of_tunnels):
    for tunnel_in_list in list_of_tunnels:
        assert isinstance(tunnel_in_list, Tunnel)
        if tunnel._nodes[-1] in tunnel_in_list._nodes:
            return True
        elif tunnel._nodes[0] in tunnel_in_list._nodes:
            return True
    return False

def main():
    # Generate the vertices of the mesh
    with open("graph.pkl", "rb") as f:
        graph = pickle.load(f)
    # Order the tunnels so that the meshes intersect
    assert isinstance(graph, TunnelNetwork)
    tunnels_to_assign = list(graph._tunnels.copy())

    assigned_tunnels = [tunnels_to_assign[0],]
    tunnels_to_assign.remove(assigned_tunnels[0])

    n_tunnels = len(assigned_tunnels)
    print(f"ORDERING {n_tunnels} TUNNELS")
    while len(tunnels_to_assign) > 0:
        print(f"{len(tunnels_to_assign)}", end="/")
        for tunnel_to_assign in tunnels_to_assign:
            if tunnel_interesects_with_list(tunnel_to_assign,assigned_tunnels):
                assigned_tunnels.append(tunnel_to_assign)
                tunnels_to_assign.remove(tunnel_to_assign)
                break
    print("")
    if not os.path.isdir(MESH_FOLDER):
        os.mkdir(MESH_FOLDER)
    else:
        shutil.rmtree(MESH_FOLDER)
        os.mkdir(MESH_FOLDER)

    print("CREATING MESHES")
    for n, tunnel in enumerate(assigned_tunnels):
        print(f"{n+1} out of {n_tunnels}: ",end="Creating Pointcloud")
        vertices, normals = get_vertices_for_tunnel(tunnel)
        mesh, ptcl = mesh_from_vertices(vertices, normals)
        print(" // Creating mesh")
        o3d.io.write_triangle_mesh(f"meshes/{n}.ply", mesh)



if __name__ == "__main__":
    main()