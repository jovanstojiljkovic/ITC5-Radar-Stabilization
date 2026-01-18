import os
import time
import glob
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32, Int32

from PyQt5 import QtWidgets, QtCore, QtGui

# rotation labels
ROT_LABELS = ["0°", "90°CW", "180°", "90°CCW"]


# rotating the image, since our camera is commonly mounted sideways
def apply_rotation(bgr: np.ndarray, rot_index: int) -> np.ndarray:
    if rot_index == 1:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot_index == 2:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if rot_index == 3:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr

# ROS2 Node
class DashNode(Node):
    def __init__(self):
        super().__init__("lk_dashboard_pc")

        # Required topics
        self.image_topic   = "/rpi_camera/image_raw"
        self.pose_topic    = "/vision/lk_pose_mm"
        self.delta_topic   = "/vision/lk_delta_mm"
        self.vel_topic     = "/vision/lk_vel_mm"
        self.quality_topic = "/vision/lk_quality"
        self.valid_topic   = "/vision/lk_valid"
        self.reset_srv     = "/vision/lk_reset"

        self.last_bgr = None

        self.pose_xy  = (0.0, 0.0)
        self.delta_xy = (0.0, 0.0)
        self.vel_xy   = (0.0, 0.0)
        self.quality  = 0.0
        self.valid    = 0

        self.start_t = time.time()
        self.fps = 0.0
        self._last_frame_t = None

        self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.create_subscription(Vector3Stamped, self.pose_topic, self.on_pose, 10)
        self.create_subscription(Vector3Stamped, self.delta_topic, self.on_delta, 10)
        self.create_subscription(Vector3Stamped, self.vel_topic, self.on_vel, 10)
        self.create_subscription(Float32, self.quality_topic, self.on_quality, 10)
        self.create_subscription(Int32, self.valid_topic, self.on_valid, 10)

        self.client = self.create_client(Trigger, self.reset_srv)

    # Decode ROS Image message to BGR numpy array
    def decode_image(self, msg: Image) -> np.ndarray | None:
        
        w, h = msg.width, msg.height
        if w <= 0 or h <= 0:
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)

        enc = (msg.encoding or "").lower()

        if enc in ("bgr8", "rgb8"):
            expected = w * h * 3
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, w, 3))
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if enc in ("mono8", "8uc1"):
            expected = w * h
            if data.size < expected:
                return None
            gray = data[:expected].reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # If colors look wrong replace I420 with NV12 below.
        expected = w * h * 3 // 2
        if data.size < expected:
            return None
        yuv = data[:expected].reshape((h * 3) // 2, w)

        # Try I420
        try:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception:
            pass

        # Fallback NV12
        try:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        except Exception:
            return None

    def on_image(self, msg: Image):
        bgr = self.decode_image(msg)
        if bgr is None:
            return

        self.last_bgr = bgr

        now = time.time()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 1e-3:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst)
        self._last_frame_t = now

    def on_pose(self, msg: Vector3Stamped):
        self.pose_xy = (float(msg.vector.x), float(msg.vector.y))

    def on_delta(self, msg: Vector3Stamped):
        self.delta_xy = (float(msg.vector.x), float(msg.vector.y))

    def on_vel(self, msg: Vector3Stamped):
        self.vel_xy = (float(msg.vector.x), float(msg.vector.y))

    def on_quality(self, msg: Float32):
        self.quality = float(msg.data)

    def on_valid(self, msg: Int32):
        self.valid = int(msg.data)

    def reset(self):
        if not self.client.service_is_ready():
            self.get_logger().warn("Reset service not available.")
            return
        self.client.call_async(Trigger.Request())

# helper for loading .png logo
def load_pixmap_any(path: str) -> QtGui.QPixmap | None:
    pm = QtGui.QPixmap()
    if pm.load(path):
        return pm

    # Decode with opencv and convert to pixmap (for qt widgets)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    h, w, _ = img.shape
    qimg = QtGui.QImage(img.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg.copy())  

# QT gui window
class Window(QtWidgets.QWidget):
    def __init__(self, node: DashNode):
        super().__init__()
        self.node = node

        # Define starting rotation index
        self.rot_index = 1

        self.setWindowTitle("LK Dashboard")
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # header bar
        header = QtWidgets.QFrame()
        header.setObjectName("Header")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title = QtWidgets.QLabel("Handheld Stabilization – Live Dashboard")
        title.setObjectName("Title")

        subtitle = QtWidgets.QLabel("AAU • ITC-5-4 • Radar Stabilization")
        subtitle.setObjectName("Subtitle")

        title_box = QtWidgets.QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_wrap = QtWidgets.QWidget()
        title_wrap.setLayout(title_box)

        header_layout.addWidget(title_wrap, 1)

        # either define aau logo path as an environment variable (cmd: export AAU_LOGO_PATH=/path/to/logo.png), or give an absolute path here. Wasn't working w relative paths...
        logo_path = os.environ.get("AAU_LOGO_PATH")

        self.logo_pixmap = None
        if logo_path:
            self.logo_pixmap = load_pixmap_any(logo_path) if logo_path else None
            print("Logo load success:", self.logo_pixmap is not None, "path=", logo_path)


        self.logo_header = QtWidgets.QLabel()
        self.logo_header.setObjectName("HeaderLogo")
        self.logo_header.setFixedHeight(48)
        self.logo_header.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if self.logo_pixmap is not None:
            self.logo_header.setPixmap(self.logo_pixmap.scaledToHeight(40, QtCore.Qt.SmoothTransformation))
        else:
            self.logo_header.setText("AAU")
        header_layout.addWidget(self.logo_header, 0)

        # video display
        self.img = QtWidgets.QLabel("Waiting for image…")
        self.img.setObjectName("Video")
        self.img.setMinimumSize(480, 640)
        self.img.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.img.setAlignment(QtCore.Qt.AlignCenter)

        # sidebar - holds metrics and logo
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(320)
        s_layout = QtWidgets.QVBoxLayout(self.sidebar)
        s_layout.setContentsMargins(12, 12, 12, 12)
        s_layout.setSpacing(8)

        self.lbl_runtime = QtWidgets.QLabel("Runtime: —")
        self.lbl_fps = QtWidgets.QLabel("FPS: —")
        self.lbl_pose = QtWidgets.QLabel("Pose: x=—  y=—")
        self.lbl_delta = QtWidgets.QLabel("Delta: dx=—  dy=—")
        self.lbl_vel = QtWidgets.QLabel("Vel: vx=—  vy=—")
        self.lbl_quality = QtWidgets.QLabel("Quality: —")
        self.lbl_rot = QtWidgets.QLabel(f"Rotation: {ROT_LABELS[self.rot_index]}  (press R)")

        for w in (self.lbl_runtime, self.lbl_fps, self.lbl_pose, self.lbl_delta, self.lbl_vel, self.lbl_quality, self.lbl_rot):
            w.setObjectName("SidebarLabel")
            s_layout.addWidget(w)

        s_layout.addStretch(1)

        self.logo_sidebar = QtWidgets.QLabel()
        self.logo_sidebar.setObjectName("SidebarLogo")
        self.logo_sidebar.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom)
        self.logo_sidebar.setFixedHeight(270)
        if self.logo_pixmap is not None:
            self.logo_sidebar.setPixmap(self.logo_pixmap.scaledToWidth(270, QtCore.Qt.SmoothTransformation))
        else:
            self.logo_sidebar.setText("AAU")
        s_layout.addWidget(self.logo_sidebar)

        # Reset button + status bar
        self.btn = QtWidgets.QPushButton("Reset Optical Flow")
        self.btn.clicked.connect(self.node.reset)
        self.btn.setFixedHeight(36)

        self.status = QtWidgets.QLabel("Status: waiting for topics…")
        self.status.setObjectName("Status")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btn, 0)
        controls.addWidget(self.status, 1)

        # main layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(header)

        content = QtWidgets.QHBoxLayout()
        content.addWidget(self.img, 1)
        content.addWidget(self.sidebar, 0)

        layout.addLayout(content)
        layout.addLayout(controls)
        self.setLayout(layout)

        # styling
        self.setStyleSheet("""
            QWidget { background: #0f1115; color: #e7e9ee; font-family: 'DejaVu Sans Mono', 'Roboto Mono', monospace; }
            #Header { background: #151925; border-radius: 12px; }
            #Title { font-size: 18px; font-weight: 700; }
            #Subtitle { font-size: 12px; color: #aab0bf; }
            #Video { background: #0b0d12; border-radius: 12px; padding: 6px; }
            #Sidebar { background: #0f1318; border-radius: 8px; }
            #SidebarLabel { color: #dfe3ea; font-size: 13px; }
            #SidebarLogo { padding-top: 8px; }
            #HeaderLogo { color: #e7e9ee; font-weight: 700; }
            QPushButton {
                background: #2b6cff; border: none; border-radius: 10px;
                color: white; padding: 8px 14px; font-weight: 700;
            }
            QPushButton:hover { background: #245ee3; }
            QPushButton:pressed { background: #1f53c9; }
            #Status { color: #aab0bf; }
        """)

        # UI refresh timer - displayed fps is reduced if this is less than camera fps
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33) 

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Press R to cycle rotation
        if event.key() in (QtCore.Qt.Key_R,):
            self.rot_index = (self.rot_index + 1) % 4
            self.lbl_rot.setText(f"Rotation: {ROT_LABELS[self.rot_index]}  (press R)")
        super().keyPressEvent(event)

    def tick(self):
        frame = self.node.last_bgr
        if frame is None:
            return

        runtime = time.time() - self.node.start_t
        fps = self.node.fps

        x, y = self.node.pose_xy
        dx, dy = self.node.delta_xy
        vx, vy = self.node.vel_xy
        qual = self.node.quality

        self.lbl_runtime.setText(f"Runtime: {runtime:6.1f}s")
        self.lbl_fps.setText(f"FPS: {fps:5.1f}")
        self.lbl_pose.setText(f"Pose: x={x:7.2f}  y={y:7.2f}")
        self.lbl_delta.setText(f"Delta: dx={dx:6.2f} dy={dy:6.2f}")
        self.lbl_vel.setText(f"Vel: vx={vx:6.2f} vy={vy:6.2f}")
        self.lbl_quality.setText(f"Quality: {qual:0.3f}")

        self.status.setText(
            f"Image: {self.node.image_topic} | Pose: {self.node.pose_topic} | Reset: {self.node.reset_srv}"
        )

        bgr = apply_rotation(frame, self.rot_index)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.img.setPixmap(pix.scaled(self.img.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


def main():
    rclpy.init()
    node = DashNode()

    app = QtWidgets.QApplication([])
    win = Window(node)
    win.resize(1200, 860)
    win.show()

    spin_timer = QtCore.QTimer()
    spin_timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0.0))
    spin_timer.start(5)

    app.exec_()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
