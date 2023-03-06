#!/usr/bin/python3

import csv
import os.path
import sys

import matplotlib
import rospy
import tf
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Twist, PoseStamped
from nav_msgs.msg import Odometry
from matplotlib import pyplot as plt
from std_msgs.msg import Float32, Float32MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

matplotlib.use("Qt5Agg")
from slibon.msg import CnnResult, Circle

from PyQt5.QtWidgets import (
    QMenuBar,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QSplitter,
    QLabel,
    QGridLayout,
    QLineEdit,
    QComboBox,
    QFileDialog,
)
from EasyConfig import EasyConfig
from subt_proc_gen.tunnel import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer


class MainWindow(QtWidgets.QMainWindow):
    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)

    class Param(QWidget):
        def __init__(self, name, kind, dflt, increment):
            super().__init__()
            self.value = dflt
            self.dflt = dflt
            vbox = QVBoxLayout()
            self.setLayout(vbox)
            self.period = 0
            self.fmt = "{}"
            self.kind = kind
            self.timer = None
            self.label = QLabel(self.fmt.format(self.value))
            self.label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(self.label)
            vbox.addWidget(QLabel("Topic: " + name))
            plus = QPushButton("+")
            plus.clicked.connect(self.inc)
            minus = QPushButton("-")
            minus.clicked.connect(self.dec)
            hbox = QHBoxLayout()
            vbox.addLayout(hbox)
            hbox.addWidget(minus)
            hbox.addWidget(plus)
            self.increment = increment
            self.pub = rospy.Publisher(name, kind, queue_size=1)
            self.topic = name
            self.min = 0
            self.setContentsMargins(0, 0, 0, 0)
            self.layout().setContentsMargins(0, 0, 0, 0)
            # self.layout().setSizeConstraint(Qt.Six)

        def setFormat(self, fmt):
            self.fmt = fmt
            self.label.setText(self.fmt.format(self.value))

        def setValue(self, value):
            self.value = value
            self.label.setText(self.fmt.format(self.value))

        def inc(self):
            self.value = self.value + self.increment
            self.publish(self.value)

        def dec(self):
            self.value = max(self.min, self.value - self.increment)
            self.publish(self.value)

        def stop(self):
            self.timer.stop()
            self.timer = None

        def start(self):
            if self.period > 0 and self.timer == None:
                self.timer = QTimer()
                self.timer.timeout.connect(self.loop)
                self.timer.start(self.period)

        def setPeriod(self, ms):
            self.period = ms

        def loop(self):
            self.publish(self.value)

        def getValue(self):
            return self.value

        def getTopic(self):
            return self.topic

        def getDefault(self):
            return self.dflt

        def publish(self, value):
            self.label.setText(self.fmt.format(self.value))
            v = self.kind()
            v.data = value
            self.pub.publish(v)

    def configure(self):
        self.paramc = EasyConfig()

        for p in self.params:
            self.paramc.root().addFloat(
                p.getTopic().replace("/", "_"), default=p.getDefault()
            )

        if self.config.get("last"):
            self.load_config(self.config.get("last"))

    def load_config(self, filename):
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Open UAVC file", "", "UAVC Files (*.uavc)"
            )

        if filename is not None and filename != "":
            self.paramc.load(filename)
            for p in self.params:
                val = self.paramc.get(p.getTopic().replace("/", "_"))
                print("getting param", p.getTopic().replace("/", "_"), val)
                if val:
                    p.setValue(val)
                    QTimer.singleShot(1000, p.loop)

            self.setWindowTitle(os.path.basename(filename))

    def save_config(self, dest):
        if dest is None:
            if self.config.get("last"):
                name = self.config.get("last")
            else:
                name = "config.uavc"
            dest, _ = QFileDialog.getSaveFileName(
                self, "Save UAVC File", name, "UAVC Files (*.uavc)"
            )
        if dest and dest != "":

            for p in self.params:
                self.paramc.set(p.getTopic().replace("/", "_"), p.getValue())
            self.paramc.save(dest)
            self.config.set("last", dest)
            self.config.save("uavc.yaml")
            self.setWindowTitle(os.path.basename(dest))

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.accumulated_distance = 0
        self.prev_id = None
        rospy.init_node("uavc", anonymous=True)
        self.params = []

        self.config = EasyConfig()
        self.config.root().addFolderChoice("models_folder", "Models Directory")

        ss = self.config.root().addHidden("hidden")
        ss.addString("current_model")
        ss.addString("last")

        self.config.load("uavc.yaml")

        menu = QMenuBar()
        m1 = menu.addMenu("File")
        m1.addAction("Load Config", lambda: self.load_config(None))
        m1.addAction("Save Config", self.save_config)
        m2 = menu.addMenu("Edit")
        m2.addAction("Config", self.edit)

        self.setMenuBar(menu)
        self.lay = QSplitter()
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)

        set_btn = QPushButton("Set")
        self.p1 = self.Param("/slibon/command_x", Float32, 0, 0.5)

        def set_model():
            self.p1.setValue(0)

            m = ModelState()
            m.pose.position.x = float(self.model_poses[0].text())
            m.pose.position.y = float(self.model_poses[1].text())
            m.pose.position.z = float(self.model_poses[2].text())
            q = quaternion_from_euler(
                float(self.model_poses[3].text()) * 3.1416 / 180,
                float(self.model_poses[4].text()) * 3.1416 / 180,
                float(self.model_poses[5].text()) * 3.1416 / 180,
            )

            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]

            m.model_name = self.set_model_combo.currentText()
            self.counter = 0

            def pub():
                self.counter += 1
                if self.counter > 20:
                    self.q.stop()
                self.pub.publish(m)

            self.q = QTimer()
            self.q.timeout.connect(pub)
            self.q.start(50)

            ct = self.set_model_combo.currentText()
            states: ModelStates = rospy.wait_for_message(
                "/gazebo/model_states", ModelStates
            )
            self.set_model_combo.clear()
            self.set_model_combo.addItems(states.name)
            if ct in [
                self.set_model_combo.itemText(i)
                for i in range(self.set_model_combo.count())
            ]:
                self.set_model_combo.setCurrentText(ct)

            for p in self.params:
                p.publish(p.getValue())

            current = self.models_cb.currentText()
            models = os.listdir(self.config.get("models_folder"))
            self.models_cb.clear()
            self.models_cb.addItems(models)
            self.models_cb.setCurrentText(current)

        set_btn.clicked.connect(set_model)

        vb = QVBoxLayout()

        models = os.listdir(self.config.get("models_folder"))
        self.models_cb = QComboBox()
        self.models_cb.addItems(models)

        def model_changed():
            self.config.set("current_model", self.models_cb.currentText())
            self.config.save("uavc.yaml")

        if self.config.get("current_model"):
            self.models_cb.setCurrentText(self.config.get("current_model"))

        self.models_cb.currentIndexChanged.connect(model_changed)
        label = QLabel("Model")
        label.setStyleSheet("font-weight: bold;")
        vb.addWidget(label)
        vb.addWidget(self.models_cb)
        self.addLine(vb)
        xyz = QGridLayout()

        self.model_poses = []
        for p in range(0, 6):
            le = QLineEdit("0")
            le.setMaximumWidth(40)
            self.model_poses.append(le)

        xyz.addWidget(QLabel("X"), 0, 0)
        xyz.addWidget(QLabel("Y"), 0, 1)
        xyz.addWidget(QLabel("Z"), 0, 2)
        for i in range(3):
            xyz.addWidget(self.model_poses[i], 1, i)
        xyz.addWidget(QLabel("R"), 2, 0)
        xyz.addWidget(QLabel("P"), 2, 1)
        xyz.addWidget(QLabel("Y"), 2, 2)
        for i in range(3):
            xyz.addWidget(self.model_poses[i + 3], 3, i)
        self.set_model_combo = QComboBox()

        label = QLabel("Set Pose")
        label.setStyleSheet("font-weight: bold;")
        vb.addWidget(label)

        vb.addWidget(self.set_model_combo)

        states: ModelStates = rospy.wait_for_message(
            "/gazebo/model_states", ModelStates
        )
        self.set_model_combo.addItems(states.name)

        vb.addLayout(xyz)

        hbox = QHBoxLayout()
        vb.addWidget(set_btn)
        self.addLine(vb)

        self.p1.setPeriod(100)
        self.p1.setFormat("{:.1f} m/s")
        self.p1.start()

        qb = QComboBox()

        def speed_changed():
            self.p1.setValue(float(qb.currentText()))

        qb.addItems(
            [
                "0",
                "0.25",
                "0.5",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "6.5",
                "7",
                "7.5",
                "8",
                "8.5",
            ]
        )
        qb.activated.connect(speed_changed)

        label = QLabel("Speed")
        label.setStyleSheet("font-weight: bold;")
        vb.addWidget(label)

        pb = QPushButton("Go")

        pb.clicked.connect(lambda: self.p1.setValue(float(qb.currentText())))
        pb.setMaximumWidth(50)
        hb = QHBoxLayout()
        hb.addWidget(qb)
        hb.addWidget(pb)

        vb.addLayout(hb)

        vb.addWidget(self.p1)
        self.params.append(self.p1)

        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(lambda: self.p1.setValue(0))
        vb.addWidget(stop_btn)

        self.addLine(vb)

        p2 = self.Param("/slibon/mult_y", Float32, 1, 0.1)
        p2.setFormat("{:.1f}")
        vb.addWidget(p2)
        self.params.append(p2)
        self.addLine(vb)

        p2 = self.Param("/slibon/mult_z", Float32, 1, 0.1)
        p2.setFormat("{:.1f}")
        vb.addWidget(p2)
        self.addLine(vb)
        self.params.append(p2)

        p2 = self.Param("/slibon/mult_w", Float32, 4, 0.1)
        p2.setFormat("{:.1f}")
        vb.addWidget(p2)
        self.params.append(p2)

        self.addLine(vb)

        p3 = self.Param("/slibon/mult_safety", Float32, 1, 0.1)
        p3.setFormat("{:.1f}")
        vb.addWidget(p3)
        self.params.append(p3)
        """
        p3 = self.Param("/slibon/pid/kp", Float32, 1, 0.1)
        p3.setFormat("{:.1f}")
        vb.addWidget(p3)
        self.params.append(p3)

        p3 = self.Param("/slibon/pid/kd", Float32, 0, 0.1)
        p3.setFormat("{:.1f}")
        vb.addWidget(p3)
        self.params.append(p3)

        p3 = self.Param("/slibon/pid/ki", Float32, 0, 0.1)
        p3.setFormat("{:.1f}")
        vb.addWidget(p3)
        self.params.append(p3)
        """
        self.addLine(vb)

        self.configure()

        # SPLINE ##############################
        def move_spline():
            if self.sender().isChecked():
                folder = (
                    self.config.get("models_folder")
                    + os.sep
                    + self.config.get("current_model")
                )
                print(folder)
                if folder:
                    poses = np.loadtxt(folder + os.sep + "spline.csv")
                    self.a = QTimer()
                    self.i = 0

                    def do_move():
                        m = ModelState()
                        m.model_name = "quadrotor"

                        vector = (poses[self.i, 4], poses[self.i, 5])
                        perp = [-poses[self.i, 5], poses[self.i, 4]]

                        angle = np.arctan2(vector[1], vector[0])

                        print("I:", self.i)
                        if self.i > 3750:
                            dy = math.cos(self.i * 0.025)
                            m.pose.position.x = poses[self.i, 1] + perp[0] * dy
                            m.pose.position.y = poses[self.i, 2] + perp[1] * dy
                        else:
                            m.pose.position.x = poses[self.i, 1]
                            m.pose.position.y = poses[self.i, 2]

                        m.pose.position.z = poses[self.i, 3] + 0.5

                        dyaw = math.cos(self.i * 0.05 / 4) * 40
                        if self.i > 2500:
                            droll = math.cos(self.i * 0.025 / 4 + 0.4) * 20
                        else:
                            droll = 0

                        if self.i > 1250:
                            dpitch = math.cos(self.i * 0.07 / 4 + 1) * 20
                        else:
                            dpitch = 0

                        q = quaternion_from_euler(
                            droll * 3.14 / 180,
                            dpitch * 3.14 / 180,
                            angle + dyaw * 3.14 / 180,
                        )
                        m.pose.orientation.x = q[0]
                        m.pose.orientation.y = q[1]
                        m.pose.orientation.z = q[2]
                        m.pose.orientation.w = q[3]
                        #                        print(m.pose.position.x)

                        if self.record_btn.isChecked():
                            self.i = self.i + 1

                        if self.i >= len(poses):
                            self.i = 0

                        self.pub.publish(m)
                        self.moved = 50

                    self.a.timeout.connect(lambda: do_move())
                    self.a.start(100)
            else:
                self.a.stop()

        self.move_spline_btn = QPushButton("Move Spline")
        self.move_spline_btn.clicked.connect(move_spline)
        self.move_spline_btn.setCheckable(True)

        vb.addWidget(self.move_spline_btn)
        self.addLine(vb)

        # POSES ##############################

        def record_pose():
            model_folder = (
                self.config.get("models_folder")
                + os.sep
                + self.config.get("current_model")
            )

            if not self.record_btn.isChecked():
                self.cnn_sub.unregister()
                self.f.close()
                self.accumulated_distance = 0
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Open csv file",
                    model_folder + os.sep + "poses.csv",
                    "Pose Files (*.csv)",
                )

                if filename is not None and filename != "":
                    os.rename(model_folder + os.sep + "poses.csv", filename)

                self.save_config(filename.replace(".csv", ".uavc"))

            else:
                self.accumulated_distance = 0
                self.prev_id = None

                self.f = open(model_folder + os.sep + "poses.csv", "w")
                spline = np.loadtxt(model_folder + os.sep + "spline.csv")
                self.listener = tf.TransformListener()

                def callback_cnn(angle: CnnResult):
                    print(self.moved)

                    self.moved = self.moved - 1

                    if self.moved > 48:
                        return

                    data = rospy.wait_for_message("/gazebo/model_states", ModelStates)

                    speed: Odometry = rospy.wait_for_message(
                        "/ground_truth/state", Odometry
                    )

                    pose: Pose = data.pose[0]
                    r, p, y = euler_from_quaternion(
                        [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                    )

                    min_dist = 1e6
                    min_dist_id = -1

                    for i in range(len(spline)):
                        spx, spy, spyaw = (
                            spline[i, 1],
                            spline[i, 2],
                            np.arctan2(spline[i, 4], spline[i, 5]),
                        )
                        if (
                            math.sqrt(
                                math.pow(spx - pose.position.x, 2)
                                + math.pow(spy - pose.position.y, 2)
                            )
                            < min_dist
                        ):
                            min_dist = math.sqrt(
                                math.pow(spx - pose.position.x, 2)
                                + math.pow(spy - pose.position.y, 2)
                            )
                            min_dist_id = i

                    if self.prev_id is None:
                        self.prev_id = (pose.position.x, pose.position.y)

                    covered = math.sqrt(
                        math.pow(pose.position.x - self.prev_id[0], 2)
                        + math.pow(pose.position.y - self.prev_id[1], 2)
                    )
                    self.accumulated_distance = self.accumulated_distance + covered

                    self.prev_id = (pose.position.x, pose.position.y)

                    spx, spy, spz, spyaw = (
                        spline[min_dist_id, 1],
                        spline[min_dist_id, 2],
                        spline[min_dist_id, 3],
                        np.arctan2(spline[min_dist_id, 5], spline[min_dist_id, 4]),
                    )

                    msg: Circle = rospy.wait_for_message("/slibon/circle_car", Circle)
                    pa = PoseStamped()
                    pa.header.frame_id = "vertical_base_link"
                    pa.pose.position.x = msg.x
                    pa.pose.position.y = msg.y
                    pa.pose.position.z = msg.z
                    pa.header.stamp = rospy.Time(0)

                    self.listener.waitForTransform(
                        "/vertical_base_link",
                        "/world",
                        rospy.Time(0),
                        rospy.Duration(4.0),
                    )
                    pa = self.listener.transformPose("/world", pa)
                    print(pa.pose.position)

                    if not self.f.closed:
                        print("distance", covered, self.accumulated_distance)
                        self.f.write(
                            "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                                pose.position.x,
                                pose.position.y,
                                pose.position.z,
                                r,
                                p,
                                y,
                                spx,
                                spy,
                                spz,
                                spyaw,
                                angle.result,
                                self.accumulated_distance,
                                speed.twist.twist.linear.x,
                                speed.twist.twist.linear.y,
                                speed.twist.twist.angular.z,
                                pa.pose.position.x,
                                pa.pose.position.y,
                                msg.r,
                            )
                        )

                self.cnn_sub = rospy.Subscriber(
                    "/slibon/cnn_angle", CnnResult, callback_cnn, queue_size=1
                )

        self.record_btn = QPushButton("Record Pose")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(record_pose)
        vb.addWidget(self.record_btn)

        self.addLine(vb)

        plot = QPushButton("Plot")

        def plotting():
            x = []
            y = []
            yaw = []
            spx = []
            spy = []
            spyaw = []
            spcnn = []

            model_folder = (
                self.config.get("models_folder")
                + os.sep
                + self.config.get("current_model")
            )
            with open(model_folder + os.sep + "poses.csv", "r") as csvfile:

                plots = csv.reader(csvfile, delimiter=",")

                for row in plots:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
                    yaw.append(float(row[5]))
                    spx.append(float(row[6]))
                    spy.append(float(row[7]))
                    spyaw.append(float(row[9]))
                    spcnn.append(float(row[10]))

            fig, axs = plt.subplots(3, 1)
            axs[0].plot(x, y, color="r")
            axs[0].plot(spx, spy, color="b")
            axs[0].axis("equal")

            axs[1].plot(yaw, color="r")
            axs[1].plot(spyaw, color="b")

            axs[2].plot(np.convolve(spcnn, np.ones(6), "valid") / 6, color="b")
            axs[2].plot(np.array(yaw) - np.array(spyaw), color="r")

            plt.show()

        plot.clicked.connect(plotting)
        vb.addWidget(plot)

        helper = QWidget()
        helper.setLayout(vb)

        self.setCentralWidget(helper)

        self.show()

    def addLine(self, vb):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        vb.addWidget(line)

    def edit(self):
        self.config.edit()
        self.config.save("uavc.yaml")

    def current_tab_changed(self, num):
        if num == 0:
            pass
        if num == 1:
            pass
        elif num == 2:
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
