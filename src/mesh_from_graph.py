import pickle
from mesh_generation import *
import open3d as o3d
from graph import Graph
from PARAMS import *


def main():
    # Generate the vertices of the mesh
    with open("graph.pkl", "rb") as f:
        graph = pickle.load(f)
    print("Generating mesh")
    points, normals = get_vertices_for_tunnels(graph)
    print(points)
    meshes, ptcls = list(), list()
    # Get the mesh for each tunnel
    for i, (p, n) in enumerate(zip(points, normals)):
        ptcl = o3d.geometry.PointCloud()
        ptcl.points = o3d.utility.Vector3dVector(p.T)
        ptcl.normals = o3d.utility.Vector3dVector(n.T)
        ptcl.colors = o3d.utility.Vector3dVector(np.ones(np.asarray(ptcl.points).shape)*COLORS[i])
        ptcls.append(ptcl)
    axis_ptcls = list()
    for tunnel in graph._tunnels:
        axis_ptcls.append(get_axis_pointcloud(tunnel))
    o3d.visualization.draw_geometries(ptcls+axis_ptcls)



if __name__ == "__main__":
    main()