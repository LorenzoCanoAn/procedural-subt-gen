import os
import sys

import matplotlib
import pyvista
import yaml

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtGui import QPixmap, QPainter, QPen, QIcon
from PyQt5.QtWidgets import (
    QMenuBar,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFrame,
    QTabWidget,
    QSplitter,
    QLabel,
    QFileDialog,
    QToolBar,
    QInputDialog,
    QMenu,
    QSlider,
    QMessageBox,
    QScrollArea,
    QProgressDialog,
)
from pyvista import QtInteractor
from pyvistaqt import QtInteractor
from EasyConfig import EasyConfig
from pointcloud_from_graph import pc_from_graph
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.tunnel import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QRunnable, QThreadPool, pyqtSignal, QObject


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Point(QPoint):
    def __init__(self, pose=None, index=0):
        super().__init__()
        if pose is not None:
            self.setX(pose.x())
            self.setY(pose.y())
            self.index = index
            self.zpos = 0

    def z(self):
        return self.zpos

    def setZ(self, val):
        self.zpos = val

    def nppose(self):
        return np.array([self.x(), self.y(), self.z()])

    def serialize(self, dictionary):
        dictionary["index"] = self.index
        dictionary["x"] = self.x()
        dictionary["y"] = self.y()
        dictionary["z"] = self.z()

    def deserialize(self, dictionary):
        self.index = dictionary["index"]
        self.setX(dictionary["x"])
        self.setY(dictionary["y"])
        self.setZ(dictionary["z"])


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
        self.current_tree = []
        self.selected = None

    def delete_last(self):
        if len(self.current_tree) > 0:
            point = self.current_tree.pop(-1)
            found = False
            for tree, color in self.trees:
                found = found or (point in tree)
            if not found:
                self.points.remove(point)
                for i, node in enumerate(self.points):
                    node.index = i

            if len(self.current_tree) > 1:
                self.p1 = self.current_tree[-1]
            else:
                self.current_tree = []
                self.p1 = None
                self.p2 = None

            self.update()

    def load(self, filename):

        self.points.clear()
        self.current_tree.clear()
        self.trees.clear()
        try:
            f = open(filename, "r")
            data = yaml.safe_load(f)
            f.close()

            if len(data["points"]) > 0:
                for p in data["points"]:
                    q = Point()
                    q.deserialize(p)
                    self.points.append(q)

            for color, nodes in enumerate(data["trees"]):
                tree = []
                for node_id in nodes:
                    tree.append(self.points[node_id])
                self.trees.append((tree, color))

            self.update()
            return True
        except:
            return False

    def setScale(self, scale):
        self.scale = scale

    def clear_points(self):
        self.points.clear()
        self.trees.clear()
        self.current_tree.clear()
        self.p1 = None
        self.p2 = None
        self.color_index = 0
        self.update()

    def save(self, filename):
        dictionary = {"points": [], "trees": []}

        if len(self.current_tree) > 0 and self.current_tree not in [
            t for t, c in self.trees
        ]:
            self.trees.append((self.current_tree, self.color_index))
            self.current_tree = []

        for p in self.points:
            data = {}
            p.serialize(data)
            dictionary["points"].append(data)

        for tree, color in self.trees:
            dictionary["trees"].append([node.index for node in tree])
        with open(filename, "w") as f:
            yaml.dump(dictionary, f)

    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        for p in self.points:
            r = QRect(p.x() - 10, p.y() - 10, 20, 20)
            if r.contains(ev.pos().x(), ev.pos().y()):
                qm = QMenu()
                set_z, cont = qm.addAction("Set Z"), None
                for tree, color in self.trees:
                    if p == tree[-1]:
                        cont = qm.addAction("Continue here")
                        break

                pos = self.mapToGlobal(ev.pos())
                res = qm.exec(pos)

                if res == cont:
                    self.p1 = p
                    self.current_tree = tree
                    self.color_index = color
                    return

                elif res == set_z:
                    text, ok = QInputDialog.getText(
                        self,
                        "Set Z coordinate of node " + str(p.index),
                        "Value",
                        text=str(p.z()),
                    )
                    if ok and text.replace(".", "").replace("-", "").isnumeric():
                        p.setZ(float(text))
                        self.update()

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(ev)
        for p in self.points:
            r = QRect(p.x() - 10, p.y() - 10, 20, 20)
            if r.contains(ev.pos().x(), ev.pos().y()):
                text, ok = QInputDialog.getText(
                    self,
                    "Set Z coordinate of node " + str(p.index),
                    "Value",
                    text=str(p.z()),
                )
                if ok and text.replace(".", "").replace("-", "").isnumeric():
                    p.setZ(float(text))
                    self.update()
                break

    def getPoints(self, scale):

        if len(self.current_tree) > 0 and self.current_tree not in [
            t for t, c in self.trees
        ]:
            self.trees.append((self.current_tree, self.color_index))
            self.current_tree = []

        self.p1 = None
        self.p2 = None
        self.update()

        if len(self.points) == 0:
            return

        nodes = []
        for p in self.points:
            nodes.append(CCaveNode(p.nppose() * scale))

        x, y, z = nodes[0].coords
        for n in nodes:
            n.coords[0] = -x + n.coords[0]
            n.coords[1] = y - n.coords[1]
            n.coords[2] = -z + n.coords[2]

        node_trees = list()

        for tree, color in self.trees:
            node_tree = list()
            for p in tree:
                node_tree.append(nodes[p.index])
            node_trees.append(node_tree)

        return nodes, node_trees

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(ev)
        if ev.modifiers() & Qt.ControlModifier:
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    self.selected = p

        elif ev.modifiers() & Qt.ShiftModifier:
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    if len(self.current_tree) > 0 and self.current_tree not in [
                        t for t, c in self.trees
                    ]:
                        self.trees.append((self.current_tree, self.color_index))
                    self.current_tree = [p]
                    self.p1 = p
                    self.color_index = self.color_index + 1

        elif self.p1 is None and len(self.points) == 0:
            self.base = ev.pos()
            self.p1 = Point(ev.pos(), 0)
            self.current_tree.append(self.p1)
            self.points.append(self.p1)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        if self.selected:
            self.selected.setX(ev.pos().x())
            self.selected.setY(ev.pos().y())
            self.update()
        elif self.p1 is not None:
            self.p2 = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if self.selected:
            self.selected = None
        elif self.p1 and self.p2:
            found = None
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    p.setX(self.p2.x())
                    p.setY(self.p2.y())
                    found = p
                    break
            if found:
                self.current_tree.append(found)
                self.p1 = found
            else:
                self.p1 = Point(self.p2, len(self.points))
                self.current_tree.append(self.p1)
                self.points.append(self.p1)
            self.p2 = None
            self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)

        for i in range(len(self.current_tree)):
            p = self.current_tree[i]
            painter.setPen(Qt.black)
            if p.z() != 0:
                painter.drawText(
                    QPoint(p.x() + 10, p.y() + 10), str(p.index) + " z:" + str(p.z())
                )
            else:
                painter.drawText(QPoint(p.x() + 10, p.y() + 10), str(p.index))
            painter.setPen(self.colors[self.color_index])
            painter.drawEllipse(p, 10, 10)
            if i > 0:
                painter.drawLine(p, self.current_tree[i - 1])

        for tree, color in self.trees:

            for i in range(len(tree)):
                p = tree[i]
                painter.setPen(Qt.black)
                if p.z() != 0:
                    painter.drawText(
                        QPoint(p.x() + 10, p.y() + 10),
                        str(p.index) + " z:" + str(p.z()),
                    )
                else:
                    painter.drawText(QPoint(p.x() + 10, p.y() + 10), str(p.index))

                painter.setPen(self.colors[color])
                painter.drawEllipse(p, 10, 10)
                if i > 0:
                    painter.drawLine(p, tree[i - 1])

        if self.p1 and self.p2:
            painter.setPen(QPen(self.colors[self.color_index]))
            painter.drawLine(self.p1, self.p2)

        if self.p1:
            painter.setPen(QPen(self.colors[self.color_index]))
            painter.drawEllipse(self.p1, 15, 15)

        if len(self.points) > 0:
            painter.setPen(QPen(Qt.red))
            painter.drawLine(
                self.rect().x(),
                self.points[0].y(),
                self.rect().width() + self.rect().x(),
                self.points[0].y(),
            )
            painter.setPen(QPen(Qt.green))
            painter.drawLine(
                self.points[0].x(),
                self.rect().y(),
                self.points[0].x(),
                self.rect().height() + self.rect().y(),
            )


class MainWindow(QtWidgets.QMainWindow):
    model_config = '<?xml version="1.0"?>\n\
<model>\n\
  <name>$name$</name>\n\
  <version>1.0</version>\n \
  <sdf version="1.6">model.sdf</sdf>\n\
  <description>\n\
  </description>\n\
</model>'

    model_sdf = '<?xml version="1.0"?>\n\
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
        # self.sc.figure.set_figwidth(self.geometry().width())
        # self.sc.figure.set_figheight(self.geometry().height())

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
        # vb.addWidget(self.config.get_widget())
        # vb.addWidget(cb)
        # vb.addWidget(pb)
        # vb.addWidget(rend)

        self.frame = QFrame()
        self.frame2 = QFrame()
        self.plotter = QtInteractor(self.frame)
        self.plotter2 = QtInteractor(self.frame2)

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
        self.radius_slider.setTickInterval(1)

        self.floor_slider = QSlider(Qt.Horizontal)
        self.floor_slider.setValue(1)
        self.floor_slider.setMinimum(0)
        self.floor_slider.setMaximum(8)
        self.floor_slider.setTickInterval(1)

        # self.slider.setSingleStep(100)
        self.slider.valueChanged.connect(self.slider_changed)
        self.radius_slider.valueChanged.connect(self.radius_slider_changed)
        self.floor_slider.valueChanged.connect(self.floor_slider_changed)

        render_tb.addAction("Go", self.rendera).setIcon(
            QIcon.fromTheme("media-playback-start")
        )
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

        render_tb.addWidget(QLabel("Floor: "))
        self.floor_label = QLabel("1")
        render_tb.addWidget(self.floor_label)
        render_tb.addWidget(self.floor_slider)

        vlayout.addWidget(render_tb)
        vlayout.addWidget(self.plotter.interactor)

        vvlayout = QVBoxLayout()
        vvlayout.addWidget(self.plotter2.interactor)
        self.frame2.setLayout(vvlayout)

        self.frame.setLayout(vlayout)

        helper = QWidget()
        helper.setLayout(vb)

        self.lay.addWidget(helper)
        self.tab = QTabWidget()
        self.tab.currentChanged.connect(self.current_tab_changed)

        helper = QWidget()
        vbox = QVBoxLayout(helper)
        tb = QToolBar()
        # tb.setAutoFillBackground(True)
        vbox.addWidget(tb)

        sa = QScrollArea()
        sa.setWidget(self.sketch)
        self.sketch.setMinimumWidth(int(1920 * 1.5))
        self.sketch.setMinimumHeight(int(1080 * 1.5))

        vbox.addWidget(sa)
        tb.addAction("Save", lambda: self.sketch.save(self.config.get("last"))).setIcon(
            QIcon.fromTheme("document-save")
        )
        tb.addAction("Clear", self.sketch.clear_points).setIcon(
            QIcon.fromTheme("edit-clear")
        )
        tb.addSeparator()
        tb.addAction("Delete last", self.sketch.delete_last).setIcon(
            QIcon.fromTheme("edit-undo")
        )
        tb.addSeparator()
        label = QLabel("Scale: 1 ")
        label.setMinimumWidth(70)
        tb.addWidget(label)
        self.scale_slider = QSlider()
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(20)
        self.scale_slider.setValue(10)
        self.scale_slider.setOrientation(Qt.Horizontal)
        self.scale_slider.setMaximumWidth(200)
        self.scale_slider.valueChanged.connect(
            lambda: label.setText(
                "Ratio: {:.1f} ".format(self.scale_slider.value() / 10)
            )
        )
        tb.addWidget(self.scale_slider)

        self.tab.addTab(helper, "Sketch")
        self.graph_tab = self.tab.addTab(self.sc, "Graph")

        self.tab.addTab(self.frame, "Render")
        self.tab.setTabEnabled(2, False)

        self.tab.addTab(self.frame2, "Mesh")
        self.tab.setTabEnabled(3, False)

        self.frame = QFrame()
        self.lay.addWidget(self.tab)

        pb.clicked.connect(self.doing)
        self.setCentralWidget(self.tab)

        if self.config.get("last"):
            if self.sketch.load(self.config.get("last")):
                self.setWindowTitle(self.config.get("last"))

        self.show()

    def slider_changed(self):
        self.slider_label.setText(str(self.slider.value() / 1000) + " ")

    def radius_slider_changed(self):
        self.radius_label.setText(str(self.radius_slider.value()) + " ")

    def floor_slider_changed(self):
        self.floor_label.setText(str(self.floor_slider.value()) + " ")

    def save_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save PTG Document", self.config.get("last"), "PTG Files (*.ptg)"
        )
        if dest and dest != "":
            if not dest.endswith(".ptg"):
                dest = dest + ".ptg"
            self.config.set("last", dest)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)
            self.config.save("gst.yaml")

    def new_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save PTG Document", "./tunnel.ptg", "PTG Files (*.ptg)"
        )
        if dest and dest != "":
            if not dest.endswith(".ptg"):
                dest = dest + ".ptg"
            self.sketch.clear_points()
            self.config.set("last", dest)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)
            self.config.save("gst.yaml")

    def load_yaml(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open PTG file", "", "PTG Files (*.ptg)"
        )
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
            pass  # self.rendera()
        elif num == 3:
            mesh = pyvista.read(self.model_path + "mesh.obj")
            mesh = mesh.clip("x", invert=False, origin=(2, 0, 0))

            self.plotter2.remove_all_lights()
            pyvista.global_theme.color = "red"
            self.plotter2.clear()
            self.plotter2.enable_shadows()
            self.plotter2.set_background(color="w")
            self.plotter2.add_mesh(
                mesh, show_edges=True
            )  # ,     ambient=0.2,  diffuse=0.5,    specular=0.5,    specular_power=90,)
            # self.plotter2.disable_anti_aliasing()

    def doing(self):
        if not self.config.get("last"):
            msgBox = QMessageBox()
            msgBox.setText("Please save the graph first")
            msgBox.exec()
            self.tab.setCurrentIndex(0)
            return

        #        isExist = os.path.exists(models_base_dir)
        # if not isExist:
        self.model_name = os.path.basename(os.path.splitext(self.config.get("last"))[0])
        self.model_path = (
            self.config.get("models_base_dir") + os.sep + self.model_name + os.sep
        )
        os.makedirs(self.model_path, exist_ok=True)

        self.fig.clear()
        self.graph = self.maino(self.fig, self.config)
        if self.graph:
            self.fig.canvas.draw()
            self.tab.setTabEnabled(2, True)

    def rendera(self):

        if not (self.config.get("models_base_dir")):
            msgBox = QMessageBox()
            msgBox.setText("Please set models base dir first")
            self.tab.setCurrentIndex(0)
            msgBox.exec()
            return

        self.plotter.clear()
        window = self

        class Runn(QRunnable):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def run(self) -> None:
                self.callback()

        self.tab.setCurrentIndex(2)
        self.pd = QProgressDialog(self)
        self.pd.setMaximum(0)
        self.pd.setCancelButton(None)
        self.pd.setLabelText("Rendering")
        self.pd.setModal(True)
        self.pd.show()
        if self.slider.value() == 1:
            roughness = 0.000001
        else:
            roughness = self.slider.value() / 1000

        def process():
            meshing_params = {
                "roughness": roughness,
                "radius": self.radius_slider.value(),
                "floor_to_axis_distance": self.floor_slider.value(),
            }
            proj_points, proj_normals = pc_from_graph(
                self.plotter,
                roughness,
                self.graph,
                window.model_path + "mesh.obj",
                radius=self.radius_slider.value(),
                meshing_params=meshing_params,
            )
            f = open(window.model_path + "model.config", "w")
            f.write(window.model_config.replace("$name$", window.model_name))
            f.close()
            f = open(window.model_path + "model.sdf", "w")
            f.write(window.model_sdf.replace("$name$", window.model_name))
            f.close()
            np.savetxt(window.model_path + "contour.csv", proj_points)
            # self.helper.done.emit()
            self.tab.setTabEnabled(3, True)
            self.pd.hide()

        run = Runn(process)
        #        run.helper.done.connect(lambda: self.pd.hide())
        QThreadPool.globalInstance().start(run)

    def maino(self, canvas, c, poses=None):
        for i in range(1):
            # Generate the graph
            graph = TunnelNetwork()
            central_node = CaveNode()

            tunnel_params = TunnelParams(
                {
                    "distance": np.random.uniform(c.get("min_dist"), c.get("max_dist")),
                    "starting_direction": np.array(
                        (c.get("sd_x"), c.get("sd_y"), c.get("sd_z"))
                    ),
                    "horizontal_tendency": np.deg2rad(c.get("h_tend")),
                    "horizontal_noise": np.deg2rad(c.get("h_noise")),
                    "vertical_tendency": np.deg2rad(c.get("v_tend")),
                    "vertical_noise": np.deg2rad(c.get("v_noise")),
                    "segment_length": c.get("s_length"),
                    "segment_length_noise": c.get("s_noise"),
                    "node_position_noise": c.get("np_noise"),
                }
            )

            points, trees = self.sketch.getPoints(self.scale_slider.value() / 10 / 10)
            for p in points:
                graph.add_node(p)

            gds = np.array([])
            gps = np.empty([0, 3])
            gvs = np.empty([0, 3])
            for n, t in enumerate(trees):
                tunnel1 = Tunnel(graph, params=tunnel_params)
                tunnel1.set_nodes(t)

                ds, ps, vs = tunnel1.spline.discretize(0.05)

                gds = np.concatenate((gds, ds))
                gps = np.concatenate((gps, ps))
                gvs = np.concatenate((gvs, vs))

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
