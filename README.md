# Procedural Environments for Underground Robotics
## Overview
This package contains a simple API and a GUI tool to generate underground environments in a procedural way. It was created with robotic testing in mind, so the final product is an `.obj` file.

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
If there is a need to generate more complex networks in a programatic, an API is provided.