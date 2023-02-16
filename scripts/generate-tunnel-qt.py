import os
import pickle
import sys

import numpy as np
import rospy
import matplotlib
import yaml

from mesh_generation import TunnelMeshingParams

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtGui import QPixmap, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QMenuBar, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFrame, QTabWidget, QSplitter, \
    QLabel, QFileDialog, QToolBar, QInputDialog, QMenu, QSlider, QMessageBox
from pyvista import QtInteractor
from pyvistaqt import QtInteractor
from EasyConfig import EasyConfig
from pointcloud_from_graph import pc_from_graph
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QRunnable, QThreadPool


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Point(QPoint):
    def __init__(self, pose=None, index=0, parent=-1, newtree=False, color=0):
        super().__init__()
        if pose is not None:
            self.setX(pose.x())
            self.setY(pose.y())
            self.index = index
            self.parent = parent
            self.newtree = newtree
            self.color = color
            self.zpos = 0

    def z(self):
        return self.zpos

    def setZ(self, val):
        self.zpos = val

    def nppose(self):
        return np.array([self.x() / 10, self.y() / 10, self.z() / 10])

    def serialize(self, dictionary):
        dictionary['index'] = self.index
        dictionary['color'] = self.color
        dictionary['parent'] = self.parent
        dictionary['x'] = self.x()
        dictionary['y'] = self.y()
        dictionary['z'] = self.z()

    def deserialize(self, dictionary):
        self.index = dictionary['index']
        self.color = dictionary['color']
        self.parent = dictionary['parent']
        self.setX(dictionary['x'])
        self.setY(dictionary['y'])
        self.setZ(dictionary['z'])


class CCaveNode(CaveNode):

    def __init__(self, coords=np.zeros(3)):
        self._connected_nodes = set()
        self.coords = coords
        self._tunnels = list()

class Sketch(QLabel):

    def __init__(self):
        super().__init__()
        pix = QPixmap(200, 200)
        pix.fill(Qt.white)
        self.color_index = 0
        self.colors = [Qt.red, Qt.green, Qt.blue, Qt.darkYellow, Qt.darkMagenta]
        self.setPixmap(pix)
        self.setScaledContents(True)
        self.p1 = None
        self.base = None
        self.p2 = None
        self.points = []
        self.lines = []
        self.current_tree = []
        self.trees = []
        self.scale = 0.1

    def delete_last(self):
        self.points.pop(-1)
        self.p1 = self.points[-1]
        self.update()


    def load(self, filename):

        f = open(filename, "r")
        data = yaml.safe_load(f)
        f.close()
        self.points.clear()

        if len(data['points']) > 0:
            for p in data['points']:
                q = Point()
                q.deserialize(p)
                self.points.append(q)

            self.color_index = max([p.color for p in self.points])

        self.update()

    def setScale(self, scale):
        self.scale = scale

    def clear_points(self):
        self.points.clear()
        self.p1 = None
        self.p2 = None
        self.color_index = 0
        self.update()

    def save(self, filename):
        dictionary = {'points': []}

        for p in self.points:
            data = {}
            p.serialize(data)
            dictionary['points'].append(data)

        f = open(filename, "w")
        yaml.dump(dictionary, f)


    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        for p in self.points:
            r = QRect(p.x() - 10, p.y() - 10, 20, 20)
            if r.contains(ev.pos().x(), ev.pos().y()):
                qm = QMenu()
                set_z, cont = qm.addAction("Set Z"), None

                colors = set([p.color for p in self.points])
                for c in colors:
                    p_of_this_tree = [p  for p in self.points if p.color == c]
                    if p == p_of_this_tree[-1]:
                        cont = qm.addAction("Continue here")

                pos = self.mapToGlobal(ev.pos())
                res = qm.exec(pos)
                if res == cont:
                    self.p1 = p
                    self.color_index = p.color
                    return
                elif res == set_z:
                    text, ok = QInputDialog.getText(self, 'Set Z coordinate of node '+str(p.index), 'Value', text=str(p.z()))
                    if ok and text.replace(".","").isnumeric():
                        p.setZ(float(text))



    def getPoints(self):
        if len(self.points) == 0:
            return
        nodes = []
        for p in self.points:
            nodes.append(CCaveNode(p.nppose()))


        x, y, z = nodes[0].coords
        for n in nodes:
            n.coords[0] = -x + n.coords[0]
            n.coords[1] =  y - n.coords[1]
            n.coords[2] = -z + n.coords[2]

        for p in self.points:
            if p.parent != -1:
                nodes[p.index].connect(nodes[p.parent])
        colors = set([p.color for p in self.points])
        node_trees = []
        for c in colors:

            point_tree = [p for p in self.points if p.color == c]
            if point_tree[0].parent != -1:
                point_tree.insert(0, self.points[point_tree[0].parent])

            node_tree = []
            for p in point_tree:
                node_tree.append(nodes[p.index])
            node_trees.append(node_tree)

        return nodes, node_trees

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(ev)
        if ev.modifiers() & Qt.ShiftModifier:
            self.p1 = None
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    self.p1 = p
                    self.color_index = self.color_index + 1

        elif self.p1 is None and len(self.points) == 0:
            self.base = ev.pos()
            self.p1 = Point(ev.pos(), 0)
            self.points.append(self.p1)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        if ev.modifiers() & Qt.ControlModifier:
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    p.setX(ev.pos().x())
                    p.setY(ev.pos().y())
                    self.update()


        elif self.p1 is not None:
            self.p2 = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)

        if self.p1 and self.p2:
            self.p1 = Point(self.p2, len(self.points), self.p1.index, color=self.color_index)
            self.points.append(self.p1)
            self.p2 = None
            self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        for p in self.points:
            painter.setPen(QPen(Qt.black))
            painter.drawEllipse(p, 10, 10)
            painter.drawText(p, str(p.index))
            if p.parent >= 0:
                painter.setPen(QPen(self.colors[p.color]))
                painter.drawLine(p, self.points[p.parent])

        if self.p1 and self.p2:
            painter.setPen(QPen(self.colors[self.color_index]))
            painter.drawLine(self.p1, self.p2)

        if len(self.points) > 0:
            painter.setPen(QPen(Qt.red))
            painter.drawLine(self.rect().x(),self.points[0].y(), self.rect().width() + self.rect().x(), self.points[0].y())
            painter.setPen(QPen(Qt.green))
            painter.drawLine(self.points[0].x(),self.rect().y(), self.points[0].x(), self.rect().height() + self.rect().y())


class MainWindow(QtWidgets.QMainWindow):

    model_config = \
'<?xml version="1.0"?>\n\
<model>\n\
  <name>$name$</name>\n\
  <version>1.0</version>\n \
  <sdf version="1.6">model.sdf</sdf>\n\
  <description>\n\
  </description>\n\
</model>'

    model_sdf=\
'<?xml version="1.0"?>\n\
<sdf version="1.6">\n\
  <model name="$name$">\n\
    <static>true</static>\n\
    <link name="link">\n\
      <pose>0 0 0 0 0 0</pose>\n\
      <collision name="collision">\n\
        <geometry>\n\
          <mesh>\n\
            <uri>./mesh.obj</uri>\n\
          </mesh>\n\
        </geometry>\n\
      </collision>\n\
      <visual name="visual">\n\
        <geometry>\n\
          <mesh>\n\
            <uri>./mesh.obj</uri>\n\
          </mesh>\n\
        </geometry>\n\
      </visual>\n\
    </link>\n\
  </model>\n\
</sdf>'


    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)
        self.sc.figure.set_figwidth(self.geometry().width())
        self.sc.figure.set_figheight(self.geometry().height())

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.config = EasyConfig()
        random_gen = self.config.root().addSubSection("Random Generation")

        dist = random_gen.addSubSection("Distance")
        dist.addInt("min_dist", "Min", 1)
        dist.addInt("max_dist", "Max", 10)

        dist = random_gen.addSubSection("Starting direction")
        dist.addInt("sd_x", "X", 1)
        dist.addInt("sd_y", "Y", 0)
        dist.addInt("sd_z", "Z", 0)

        dist = random_gen.addSubSection("Horizontal")
        dist.addInt("h_tend", "Tendency", 1)
        dist.addInt("h_noise", "Noise", 0)

        dist = random_gen.addSubSection("Vertical")
        dist.addInt("v_tend", "Tendency", 0)
        dist.addInt("v_noise", "Noise", 0)

        dist = random_gen.addSubSection("Segment")
        dist.addFloat("s_length", "Length", 10)
        dist.addInt("s_noise", "Noise", 0)
        dist.addFloat("scale", "Scale", 0.1)
        dist.addFloat("roughness", "Roughness", 0.0001)
        random_gen.addInt("np_noise", "Node Position Noise")

        models = self.config.root().addSubSection("Models")
        models.addFolderChoice("models_base_dir", "Base folder")

        private = self.config.root().addSubSection("@Private")
        private.addString("last")



        self.config.load("gst.yaml")

        menu = QMenuBar()
        m1 = menu.addMenu("File")
        m1.addAction("Load", self.load_yaml)
        m1.addAction("New", self.new_yaml)
        m1.addAction("Save as ", self.save_yaml)

        m2 = menu.addMenu("Edit")
        m2.addAction("Config", self.edit)

        self.setMenuBar(menu)
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.fig = Figure()  # figsize=[200,200]
        self.sc = FigureCanvasQTAgg(self.fig)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.lay = QSplitter()
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)
        # helper = QWidget()
        # helper.setLayout(self.lay)
        self.sketch = Sketch()

        cb = QPushButton("Clear")
        cb.clicked.connect(self.sketch.clear_points)
        pb = QPushButton("Graph")
        vb = QVBoxLayout()
        rend = QPushButton("Render")
        rend.clicked.connect(self.rendera)
        #vb.addWidget(self.config.get_widget())
        #vb.addWidget(cb)
        #vb.addWidget(pb)
        #vb.addWidget(rend)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)
        vlayout = QVBoxLayout()

        render_tb = QToolBar()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(100)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setValue(4)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(8)
        self.radius_slider.setTickInterval(0.5)

        #self.slider.setSingleStep(100)
        self.slider.valueChanged.connect(self.slider_changed)
        self.radius_slider.valueChanged.connect(self.radius_slider_changed)
        render_tb.addAction("Go", self.rendera).setIcon(QIcon.fromTheme('media-playback-start'))
        render_tb.addSeparator()

        self.slider_label = QLabel("0.001")
        render_tb.addWidget(QLabel("Rougness: "))
        render_tb.addWidget(self.slider_label)
        render_tb.addWidget(self.slider)

        render_tb.addSeparator()
        self.radius_label = QLabel("4")
        render_tb.addWidget(QLabel("Radius: "))
        render_tb.addWidget(self.radius_label)
        render_tb.addWidget(self.radius_slider)
        render_tb.addSeparator()

        vlayout.addWidget(render_tb)
        vlayout.addWidget(self.plotter.interactor)

        self.frame.setLayout(vlayout)

        helper = QWidget()
        helper.setLayout(vb)

        self.lay.addWidget(helper)
        self.tab = QTabWidget()
        self.tab.currentChanged.connect(self.current_tab_changed)

        helper = QWidget()
        vbox = QVBoxLayout(helper)
        tb = QToolBar()
        #tb.setAutoFillBackground(True)
        vbox.addWidget(tb)
        vbox.addWidget(self.sketch)
        tb.addAction("Clear", self.sketch.clear_points).setIcon(QIcon.fromTheme('edit-clear'))
        tb.addAction("Delete last", self.sketch.delete_last).setIcon(QIcon.fromTheme('edit-undo'))

        self.tab.addTab(helper, "Sketch")
        self.graph_tab = self.tab.addTab(self.sc, "Graph")
        self.tab.addTab(self.frame, "Render")
        self.tab.setTabEnabled(2, False)

        self.frame = QFrame()
        self.lay.addWidget(self.tab)

        pb.clicked.connect(self.doing)
        self.setCentralWidget(self.tab)

        if self.config.get("last"):
            print(self.config.get("last"))
            self.sketch.load(self.config.get("last"))
            self.setWindowTitle(self.config.get("last"))

        self.show()
    def slider_changed(self):
        self.slider_label.setText(str(self.slider.value()/1000)+" ")

    def radius_slider_changed(self):
        self.radius_label.setText(str(self.radius_slider.value())+" ")

    def save_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save YAML Document", "./tree.yaml",
                                              "YAML Files (*.yaml)")
        if dest and dest != "":
            self.config.set("last", dest)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)
            self.config.save("gst.yaml")


    def new_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save YAML Document", "./tree.yaml",
                                              "YAML Files (*.yaml)")
        if dest and dest != "":
            self.sketch.clear_points()
            self.config.set("last", dest)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)
            self.config.save("gst.yaml")


    def load_yaml(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open YAML file", "",
                                              "YAML Files (*.yaml)")
        if file is not None and file != "":
            self.config.set("last", file)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.load(file)
            self.config.save("gst.yaml")

    def edit(self):
        self.config.edit()
        self.config.save("gst.yaml")

    def current_tab_changed(self, num):
        if num == 0:
            self.sketch.setScale(self.config.get("scale"))
        if num == 1:
            self.doing()
        elif num == 2:
            self.rendera()

    def doing(self):
        if not self.config.get("last"):
            msgBox = QMessageBox()
            msgBox.setText("Please save the graph first")
            msgBox.exec()
            self.tab.setCurrentIndex(0)
            return
        if not (self.config.get("models_base_dir")):
            msgBox = QMessageBox()
            msgBox.setText("Please set models base dir first")
            self.tab.setCurrentIndex(0)
            msgBox.exec()
            return


#        isExist = os.path.exists(models_base_dir)
        #if not isExist:
        self.model_name = os.path.basename(os.path.splitext(self.config.get("last"))[0])
        self.model_path = self.config.get("models_base_dir") + os.sep + self.model_name + os.sep
        os.makedirs(self.model_path, exist_ok=True)

        self.fig.clear()
        self.graph = self.maino(self.fig, self.config)
        if self.graph:
            self.fig.canvas.draw()
            self.tab.setTabEnabled(2, True)

    def rendera(self):
        self.plotter.clear()
        window = self

        class Runn(QRunnable):
            def __init__(self, plotter, graph, roughness, radius):
                super().__init__()
                self.plotter = plotter
                self.graph = graph
                self.roughness = roughness
                self.radius = radius

            def run(self) -> None:
                pc_from_graph(self.plotter, self.roughness, self.graph, window.model_path + "mesh.obj", radius=self.radius)
                f = open(window.model_path + "model.config", "w")
                f.write(window.model_config.replace("$name$", window.model_name))
                f.close()
                f = open(window.model_path + "model.sdf", "w")
                f.write(window.model_sdf.replace("$name$", window.model_name))
                f.close()


        self.tab.setCurrentIndex(2)
        QThreadPool.globalInstance().start(Runn(self.plotter, self.graph, self.slider.value()/1000, self.radius_slider.value()))

    def maino(self, canvas, c, poses=None):
        for i in range(1):
            print("doing")
            # Generate the graph
            graph = TunnelNetwork()
            central_node = CaveNode()

            tunnel_params = TunnelParams(
                {
                    "distance": np.random.uniform(c.get("min_dist"), c.get("max_dist")),
                    "starting_direction": np.array((c.get("sd_x"), c.get("sd_y"), c.get("sd_z"))),
                    "horizontal_tendency": np.deg2rad(c.get("h_tend")),
                    "horizontal_noise": np.deg2rad(c.get("h_noise")),
                    "vertical_tendency": np.deg2rad(c.get("v_tend")),
                    "vertical_noise": np.deg2rad(c.get("v_noise")),
                    "segment_length": c.get("s_length"),
                    "segment_length_noise": c.get("s_noise"),
                    "node_position_noise": c.get("np_noise"),
                }
            )

            points, trees = self.sketch.getPoints()
            for p in points:
                graph.add_node(p)

            gds = np.array([])
            gps = np.empty([0, 3])
            gvs = np.empty([0, 3])
            for n, t in enumerate(trees):
                print("new tunnel")
                tunnel1 = Tunnel(graph, params=tunnel_params)
                tunnel1.set_nodes(t)

                ds, ps, vs = tunnel1.spline.discretize(0.05)

                gds = np.concatenate((gds, ds))
                gps = np.concatenate((gps, ps))
                gvs = np.concatenate((gvs, vs))
                print(gvs)
            gds = np.reshape(gds, [-1, 1])
            combined_spline_info = np.hstack([gds, gps, gvs])
            np.savetxt(self.model_path + os.sep + "spline.csv", combined_spline_info)
            debug_plot(graph, in_3d=True, canvas=canvas)
            return graph

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.config.save("gst.yaml")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
