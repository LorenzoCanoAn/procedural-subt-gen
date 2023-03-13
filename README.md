# Procedural Environments for Underground Robotics
## Overview
This package contains a simple API and a GUI tool to generate underground environments in a procedural way. It was created with robotic testing in mind, so the final product is an `.obj` file.

![Sketch Tab](/README_images/example_1.png)
## Instalation
After cloning the repository, move to it and run:
```
pip3 install .
```
This will make the package `subt_proc_gen` available to import.

## GUI
To create environments with a very specific topological structure, a GUI is provided that allows a user to define a tunnel network node by node, and then create a mesh based on a set of parameters. 

To open the GUI execute the script `generate-tunnel-qt.py`

### Setup
Before using the gui, it is necessary to set a folder where it will save its files, along with the resulting meshes. This folder can be set in Edit->Config->Base Folder.

### Drawing the network
In the Sketch tab there is an empty canvas where you can draw the network. The instructions to use the canvas are:

![Sketch Tab](/README_images/gui_1.png)

1. To place the first two nodes, left-click and drag. To keep adding nodes to that tunnel, keep left-clicking and dragging.
2. To start a new tunnel, press shift, left-click on a node and drag, and a new tunnel will be started from that initial node. 
3. To move a node press ctrl and left-click on a node, and drag it to its new placement.
4. If you want to extend a tunnel, right-click on one of the final nodes of a tunnel and then click "Continue Here"
5. To set the z coordinate of a node right-click on it and select "set Z".
6. On the top-left corner of the canvas there are three buttons, to save, clear the canvas and un-do the last change.
7. To create a loop, just end the click-and-drag on top on an already existent node.
8. The scale factor (sliding selector to the right of the buttons) will multiply the coordinates of every node by it, increasing the size of the final environment.

After finishing the sketch, it is necessary to save it into a file. The first time, the save locatio has to be set, so go to "File->Save As" to set the location. After this, you can save it using the save button.

### Graph Tab
The purpose of this tab is to check the final graph in 3D, and ensure that it has the desired shape.

![Graph Tab](/README_images/gui_2.png)

### Render Tab
In this tab it is possible to select the desired parameters that will control the final geometry of the mesh. Currently there are three available parameters:
1. The roughness of the tunnel walls. This will control the magnitde of the Perlin noise that will be applied to the tunnel walls.
2. Radius: This parameter controlls the base radius of all the tunnels.
3. Floor: This controls the distance from the floor to the tunnel axis. The larger the value, the lower the floor will be. If the value is greater than the radius, there will not be a flat floor.

![Graph Tab](/README_images/gui_3.png)

### Mesh Tab
This is a visualization tab for the final mesh, that has been saved into the mesh.obj file in the previously selected folder.

![Graph Tab](/README_images/gui_4.png)

## API
We also provide a high-level API to generate the networks programatically. There are two examples provided that ilustrate two ways of generating random networks by selecting the desired number of tunnels.
### Surface use
If there is no need for a very specific network structure, and the number of tunnels is the desired level of control, the following example is the simplest way of doing it.

The necessary imports are:
```
from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
import open3d as o3d
```
The parameters to adjust are:
1. The file where the mesh will be saved
2. The number of tunnels to grow from one node.
3. The number of tunnels to grow between two nodes.
4. The number of times that a tunnel generation will be tried before giving up.
```
def main():
    mesh_save_path = "/home/lorenzo/mesh.obj"
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    trial_limit = 200
```
The `TunnelNetwork` class is the one resposible of generating the network graph, and can be instanciated as is.
```
    tunnel_network = TunnelNetwork()
```
The network is initializated with a single node placed at `(0,0,0)`, so at the begining it is only possible to grow the tunnels from an initial node (as there are no final nodes). The `add_random_tunnel_from_initial_node` method chooses a random node from the graph, and creates a tunnel with random generation parameters from it. It is possible for the Tunnel generation process to be unsuccessful (if the generated tunnel collides with other tunnels already in the network). The `trial_limit` parameter controlls how many times the system will try to generate a tunnel.
```
    for _ in range(n_grown_tunnels):
        tunnel_network.add_random_tunnel_from_initial_node(trial_limit=trial_limit)
```

After the "grown" tunnels are finished, there are many more nodes in the graph, so it is possible to create tunnels between nodes. For that we use the `add_random_tunnel_from_inital_to_final_node` method. As the previous method, it will choose both the initial and final node of the tunnel randomly form the network, and then generate a tunnel that connects them.
```
    for _ in range(n_connecting_tunnels):
        tunnel_network.add_random_tunnel_from_initial_to_final_node(
            trial_limit=trial_limit
        )
```
After adding the desired number of tunnels, the `TunnelNetwork` object can be transformed into a mesh with the help of the `TunnelNetworkWithMesh` class. When creating this mesh, we can either introduce the meshing parameters that we desire (by using a `TunnelMeshingParameters` object as the `i_meshing_params` argument), or we can let them be chosen randomly, as it is the case here.
```
    tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
    tnm.clean_intersections()
    _, mesh = tnm.compute_mesh()
```
The `compute_mesh` method returns an Open3D `TriangleMesh` object, that we can save into a file with the `write_triangle_mesh` function.
```
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
```

The complete script.
```
from subt_proc_gen.tunnel import TunnelNetwork
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
import open3d as o3d

def main():
    mesh_save_path = "/home/lorenzo/mesh.obj"
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    trial_limit = 200
    tunnel_network = TunnelNetwork()
    for _ in range(n_grown_tunnels):
        tunnel_network.add_random_tunnel_from_initial_node(trial_limit=trial_limit)
    for _ in range(n_connecting_tunnels):
        tunnel_network.add_random_tunnel_from_initial_to_final_node(
            trial_limit=trial_limit
        )
    tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
    tnm.clean_intersections()
    _, mesh = tnm.compute_mesh()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)


if __name__ == "__main__":
    main()

```
### More in-depth
If there is a need to generate more tailored networks, it is possible to have more control over the generation by not using the `.add_random_tunnel_from_initial_node` and `add_random_tunnel_from_initial_to_final_node` functions. This way it is possible to select the nodes used for tunnel creation.

The following example does the same as the previous one, but shows how to directly use the `Tunnel` and `TunnelParams` classes.

Imports:

```
from subt_proc_gen.tunnel import TunnelNetwork, TunnelParams, Tunnel
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
import open3d as o3d
import random
```
The parameters are the same as in the previous example, and the initialization of the tunnel network is also the same.
```
def main():
    mesh_save_path = "/home/lorenzo/mesh.obj"
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    trial_limit = 200
    tunnel_network = TunnelNetwork()
```
The difference between this script and the one before is that the `add_random_tunnel_from_initial_node` and `add_random_tunnel_from_initial_and_final_node` methods of `tunnel_network` have not been used, but instead are implemented on the spot to show how to use the `Tunnel` class. To init a `Tunnel` it is necessary to provide it with a parent `TunnelNetwork` and a `TunnelParams`, in this case one with random values. The `Tunnel` object can then call the `compute` method with two arguments, the `initial node` (compulsory) and the `final_node` (optional). The `compute` method returns a boolean value that tells if the tunnels has been generated or not.

```
 for _ in range(n_grown_tunnels):
        # This substitutes ".add_random_tunnel_from_initial_node"
        # >>>>>>>>>
        n_trials = 0
        success = False
        while not success and trial_limit > n_trials:
            tunnel_params = TunnelParams()
            tunnel_params.random()
            tunnel = Tunnel(tunnel_network, params=tunnel_params)
            success = tunnel.compute(
                initial_node=random.choice(tunnel_network.nodes),
            )
            n_trials += 1
        # <<<<<<<<<
    for _ in range(n_connecting_tunnels):
        # This substitutes ".add_random_tunnel_from_initial_node_and_final_node"
        # >>>>>>>>>
        n_trials = 0
        success = False
        while not success and trial_limit > n_trials:
            i_node = random.choice(tunnel_network.nodes)
            f_node = random.choice(tunnel_network.nodes)
            if i_node is f_node:
                continue
            same_tunnel = False
            for tunnel in i_node.tunnels:
                if tunnel in f_node.tunnels:
                    same_tunnel = True
                    break
            if same_tunnel:
                continue
            tunnel_params = TunnelParams()
            tunnel_params.random()
            tunnel = Tunnel(
                tunnel_network,
                params=tunnel_params,
            )
            success = tunnel.compute(
                initial_node=i_node,
                final_node=f_node,
            )
            n_trials += 1
        # <<<<<<<<<
```
The point cloud generation and meshing is the same as in the previous example.
```
 tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
    tnm.clean_intersections()
    _, mesh = tnm.compute_mesh()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
```

The complete script:
```
from subt_proc_gen.tunnel import TunnelNetwork, TunnelParams, Tunnel
from subt_proc_gen.mesh_generation import TunnelNetworkWithMesh
import open3d as o3d
import random


def main():
    mesh_save_path = "/home/lorenzo/mesh.obj"
    n_grown_tunnels = 3
    n_connecting_tunnels = 2
    trial_limit = 200
    tunnel_network = TunnelNetwork()
    for _ in range(n_grown_tunnels):
        n_trials = 0
        success = False
        while not success and trial_limit > n_trials:
            tunnel_params = TunnelParams()
            tunnel_params.random()
            tunnel = Tunnel(tunnel_network, params=tunnel_params)
            success = tunnel.compute(
                initial_node=random.choice(tunnel_network.nodes),
            )
            n_trials += 1
    for _ in range(n_connecting_tunnels):
        n_trials = 0
        success = False
        while not success and trial_limit > n_trials:
            i_node = random.choice(tunnel_network.nodes)
            f_node = random.choice(tunnel_network.nodes)
            if i_node is f_node:
                continue
            same_tunnel = False
            for tunnel in i_node.tunnels:
                if tunnel in f_node.tunnels:
                    same_tunnel = True
                    break
            if same_tunnel:
                continue
            tunnel_params = TunnelParams()
            tunnel_params.random()
            tunnel = Tunnel(
                tunnel_network,
                params=tunnel_params,
            )
            success = tunnel.compute(
                initial_node=i_node,
                final_node=f_node,
            )
            n_trials += 1
    tnm = TunnelNetworkWithMesh(tunnel_network, i_meshing_params="random")
    tnm.clean_intersections()
    _, mesh = tnm.compute_mesh()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)


if __name__ == "__main__":
    main()


```