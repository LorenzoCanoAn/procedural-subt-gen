import pickle
import sys

import matplotlib
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QMenuBar, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFrame, QTabWidget, QSplitter, \
    QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from pyvista import QtInteractor
from pyvistaqt import QtInteractor
from scripts.EasyConfig import EasyConfig
from scripts.pointcloud_from_graph import pc_from_graph
from subt_proc_gen.display_functions import debug_plot
from subt_proc_gen.helper_functions import *
from subt_proc_gen.tunnel import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Sketch(QLabel):
    def __init__(self):
        super().__init__()
        pix = QPixmap(200, 200)
        pix.fill(Qt.white)
        self.setPixmap(pix)
        self.setScaledContents(True)
        self.p1 = None
        self.p2 = None
        self.points = []
        self.lines = []
    def clear_points(self):
        self.points.clear()
        self.lines.clear()
        self.update()

    def getPoints(self):
        points = []
        for p in self.points:
            print(p)
            points.append((p.x()/10, p.y()/10))
        return points


    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        if self.p1 is not None:
            self.p2 = ev.pos()
            self.update()
        pass

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(ev)
        if self.p1 is None:
            self.p1 = ev.pos()
            self.points.append(self.p1)
        pass

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if self.p1 and self.p2:
            self.lines.append((self.p1, self.p2))
            self.p1 = self.p2
            self.points.append(self.p1)
            self.p2 = None
            self.update()
        pass

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        for p in self.points:
            painter.drawEllipse(p, 5, 5)

        for l in self.lines:
            painter.drawLine(l[0], l[1])

        if self.p1 and self.p2:
            painter.drawLine(self.p1, self.p2)


class MainWindow(QtWidgets.QMainWindow):

    #    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
    #        self.sc.resize()

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.config = EasyConfig()
        dist = self.config.root().addSubSection("Distance")
        dist.addInt("min_dist", "Min", 1)
        dist.addInt("max_dist", "Max", 10)
        dist = self.config.root().addSubSection("Starting direction")
        dist.addInt("sd_x", "X", 1)
        dist.addInt("sd_y", "Y", 0)
        dist.addInt("sd_z", "Z", 0)
        dist = self.config.root().addSubSection("Horizontal")
        dist.addInt("h_tend", "Tendency", 1)
        dist.addInt("h_noise", "Noise", 0)
        dist = self.config.root().addSubSection("Vertical")
        dist.addInt("v_tend", "Tendency", 0)
        dist.addInt("v_noise", "Noise", 0)
        dist = self.config.root().addSubSection("Segment")
        dist.addInt("s_length", "Length", 10)
        dist.addInt("s_noise", "Noise", 0)
        dist = self.config.root().addInt("np_noise", "Node Position Noise")

        self.config.load("gst.yaml")

        menu = QMenuBar()
        m1 = menu.addMenu("File")
        m1.addAction("Config", self.edit)

        self.setMenuBar(menu)
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.fig = Figure()  # figsize=[200,200]
        self.sc = FigureCanvas(self.fig)
        # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        self.lay = QSplitter()
        # helper = QWidget()
        # helper.setLayout(self.lay)
        self.sketch = Sketch()

        cb = QPushButton("Clear")
        cb.clicked.connect(self.sketch.clear_points)
        pb = QPushButton("Graph")
        vb = QVBoxLayout()
        rend = QPushButton("Render")
        rend.clicked.connect(self.rendera)
        vb.addWidget(self.config.get_widget())
        vb.addWidget(cb)
        vb.addWidget(pb)
        vb.addWidget(rend)

        self.frame = QFrame()

        self.plotter = QtInteractor(self.frame)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.plotter.interactor)
        # self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        # self.lay.addWidget(self.frame)

        helper = QWidget()
        helper.setLayout(vb)

        self.lay.addWidget(helper)
        self.tab = QTabWidget()
        self.tab.addTab(self.sketch, "Sketch")
        self.graph_tab = self.tab.addTab(self.sc, "Graph")
        self.tab.addTab(self.frame, "Render")

        #        self.lay.addWidget(self.sc)
        self.frame = QFrame()
        self.lay.addWidget(self.tab)

        pb.clicked.connect(self.doing)
        # self.maino(self.fig, self.config)
        self.setCentralWidget(self.lay)

        self.show()

    def edit(self):
        self.config.edit()
        self.config.save("gst.yaml")

    def doing(self):
        self.config.root_node.collect()
        self.config.save("gst.yaml")
        # self.tab.removeTab(0)
        # self.fig = Figure(figsize=[200,200])
        # self.sc = FigureCanvas(self.fig)
        self.fig.clear()

        # self.tab.insertTab(0, self.sc, "Graph")
        # self.tab.setCurrentIndex(0)


        self.graph = self.maino(self.fig, self.config, self.sketch.getPoints())
        self.fig.canvas.draw()
        # self.sc.resize(500,500)

    def rendera(self):
        self.plotter.clear()
        pc_from_graph(self.plotter, self.graph)
        self.tab.setCurrentIndex(2)

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
            Tunnel(graph, params=tunnel_params,
                   initial_node=CaveNode(np.array([poses[0][0], poses[0][1], 0])),
                   final_node=CaveNode(np.array([poses[1][0], poses[1][1], 0])))

            for i in range(2, len(poses)):
                Tunnel(graph, params=tunnel_params,
                       initial_node=CaveNode(np.array([poses[i - 1][0], poses[i - 1][1], 0])),
                       final_node=CaveNode(np.array([poses[i][0], poses[i][1], 0])))

            '''
            node0 = CaveNode(np.array([0, 0, 0]))
            graph.add_node(node0)

            node1 = Node(np.array([10, 10, 0]))
            node1.connect(node0)
            graph.add_node(node1)

            node2 = Node(np.array([30, 30, 0]))
            node2.connect(node1)
            graph.add_node(node2)
            '''

            '''
            n0 = tunnel_0.nodes[-2]
            n1 = tunnel_0.nodes[-1]
            dir = n1.xyz - n0.xyz
            dir /= np.linalg.norm(dir)
            th, ph = vector_to_angles(dir)
            th += np.deg2rad(36)
            dir = angles_to_vector((th, ph))
            tunnel_params["starting_direction"] = dir
            tunnel_1 = Tunnel(graph, initial_node=n1, params=tunnel_params)
            # Tunnel(graph, tunnel_0.nodes[0], tunnel_0.nodes[-1], params=tunnel_params)
            '''
            debug_plot(graph, in_3d=True, canvas=canvas)

            return graph

            # plt.show()
            # with open("datafiles/graph.pkl", "wb") as f:
            #    pickle.dump(graph, f)

            # ds, ps, vs = tunnel.spline.discretize(0.05)
            # ds = np.reshape(ds, [-1, 1])
            # combined_spline_info = np.hstack([ds, ps, vs])
            # np.savetxt("datafiles/tunnel_info.txt", combined_spline_info)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
