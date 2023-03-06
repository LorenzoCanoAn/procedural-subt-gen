import os, pathlib, pickle
import pyvista as pv
import numpy as np
from subt_proc_gen.tunnel import TunnelNetwork, Tunnel, CaveNode
from subt_proc_gen.mesh_generation import (
    TunnelNetworkWithMesh,
    TunnelMeshingParams,
    TunnelWithMesh,
)
from subt_proc_gen.spline import Spline3D
import imageio
import pygifsicle
import cv2
import logging


logging.basicConfig(level=logging.DEBUG)
base_video_folder = os.path.join(
    pathlib.Path.home(),
    "Documents/my_papers/IROS2023/video/variations_on_same_environment",
)
folder_with_meshes = (
    "/home/lorenzo/git/procedural-subt-gen/datafiles/variations_on_same_environment"
)


class MyPlotter:
    def __init__(self):
        self.inactive_render = False
        self.n_captures_per_change = 2
        self.azimuth_increment = 0.4
        self.elevation_increment = 0
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background(color="white")
        self.n_image = 0
        self.n_trial = len(os.listdir(base_video_folder))
        self.images_folder = os.path.join(base_video_folder, f"trial_n_{self.n_trial}")
        print(self.images_folder)
        os.mkdir(self.images_folder)
        self.set_camera_params()
        self.images = None

    def plot_mesh(self, mesh, color, capture_at_end=True):
        actor = self.plotter.add_mesh(mesh, show_edges=True, color=color)
        if capture_at_end:
            for _ in range(self.n_captures_per_change):
                self.save_capture()
        return actor

    def save_capture(self):
        if self.inactive_render:
            return
        # path_to_image = os.path.join(self.images_folder, f"{self.n_image:04d}.png")
        path_to_image = os.path.join(self.images_folder, f"last_image.png")
        self.plotter.show(auto_close=False, screenshot=path_to_image)
        self.plotter.camera.azimuth += self.azimuth_increment
        self.plotter.camera.elevation += self.elevation_increment
        self.n_image += 1
        if self.images is None:
            self.images = [
                cv2.imread(path_to_image),
            ]
        else:
            self.images.append(cv2.imread(path_to_image))

    def print_camera_params(self):
        print(f"self.plotter.camera.position = {self.plotter.camera.position}")
        print(f"self.plotter.camera.focal_point= {self.plotter.camera.focal_point}")
        print(f"self.plotter.camera.roll = {self.plotter.camera.roll}")

    def set_camera_params(self):
        mult = 2
        self.plotter.camera.position = (
            104.5786207334545 * mult,
            111.92568096131357 * mult,
            86.94361170474676 * mult,
        )
        self.plotter.camera.focal_point = (
            -8.070702349350167,
            -0.8108855992463173,
            13.159967378983225,
        )
        self.plotter.camera.roll = -113.67940228246788

    def create_video(self):
        logging.info("Creating Video")
        path_to_video_file = os.path.join(self.images_folder, "video.avi")
        height, width, layers = self.images[0].shape
        self.video_writer = cv2.VideoWriter(path_to_video_file, 0, 24, (width, height))
        for frame in self.images:
            self.video_writer.write(frame)
        cv2.destroyAllWindows()
        self.video_writer.release()

    def clear_all(self):
        self.plotter.clear()


def main():
    my_plotter = MyPlotter()
    n_files = len(os.listdir(folder_with_meshes))
    my_plotter.plotter.camera.azimuth += 90
    for n_mesh_file, mesh_file in enumerate(os.listdir(folder_with_meshes)):
        if n_mesh_file == 70:
            break
        print(f"Mesh {n_mesh_file+1:04d} out of {n_files}", end="\r", flush=True)
        path_to_mesh = os.path.join(folder_with_meshes, mesh_file)
        mesh = pv.read(path_to_mesh)
        my_plotter.azimuth_increment = 0
        my_plotter.elevation_increment = 0
        my_plotter.clear_all()
        my_plotter.plot_mesh(mesh, color="white")
    my_plotter.create_video()


if __name__ == "__main__":
    main()
