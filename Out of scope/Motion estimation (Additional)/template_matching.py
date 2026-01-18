# Template matching node 
# Used for comparison of motion estimation methods
import sys
import threading
import time
import subprocess
import math

import cv2
import numpy as np
import serial

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32, Int32


MM_PER_PX          = 0.157566
TEMPLATE_HALF      = 50
SEARCH_HALF        = 90
TM_METHOD          = cv2.TM_CCORR_NORMED

SCORE_THRESH       = 0.65
PRE_DROP_MARGIN_PX = 12
QUALITY_POWER      = 2.5
FUSION_ALPHA       = 0.45
MIN_VALID_PATCHES  = 2
RESIDUAL_GATING_PX = 4.0
DIST_GATING_RATIO  = 0.55

PATCH_REL_OFFSETS = [
    (0.0, 0.0),
    (0.25, 0.0),
    (-0.25, 0.0),
    (0.0, 0.25),
    (0.0, -0.25),
    (0.25, 0.25),
    (-0.25, 0.25),
    (0.25, -0.25),
    (-0.25, -0.25),
]


def near_edge(top_left, tw, th, W, H, margin):
    x0, y0 = top_left
    x1, y1 = x0 + tw, y0 + th
    return (x0 <= margin) or (y0 <= margin) or (x1 >= W - margin) or (y1 >= H - margin)


def crop_template_around(gray, center_x, center_y, half, W, H):
    cx = int(round(np.clip(center_x, half, W - half)))
    cy = int(round(np.clip(center_y, half, H - half)))
    x0, y0 = cx - half, cy - half
    x1, y1 = cx + half, cy + half
    return gray[y0:y1, x0:x1].copy(), (x0, y0), (x1, y1), (cx, cy)


def match_in_roi(gray, template, expected_center, search_half, W, H, method):
    ex, ey = float(expected_center[0]), float(expected_center[1])
    cx = int(round(np.clip(ex, search_half, W - search_half)))
    cy = int(round(np.clip(ey, search_half, H - search_half)))
    x0, y0 = cx - search_half, cy - search_half
    x1, y1 = cx + search_half, cy + search_half
    roi = gray[y0:y1, x0:x1]

    res = cv2.matchTemplate(roi, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        tl_roi = min_loc
        score = float(min_val)
        quality = 1.0 - score
    else:
        tl_roi = max_loc
        score = float(max_val)
        quality = score

    top_left = (tl_roi[0] + x0, tl_roi[1] + y0)
    return top_left, score, quality


class ExponentialFusion:
    def __init__(self, alpha=0.25):
        self.alpha = float(np.clip(alpha, 1e-3, 1.0))
        self.state = None

    def reset(self):
        self.state = None

    def update(self, measurement):
        if measurement is None:
            return None if self.state is None else self.state.copy()
        measurement = np.asarray(measurement, dtype=np.float32)
        if self.state is None:
            self.state = measurement
        else:
            self.state = self.alpha * measurement + (1.0 - self.alpha) * self.state
        return self.state.copy()


class TemplateMatchingCameraNode(Node):
    def __init__(self):
        super().__init__("camera_tm_node")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 80)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = float(self.get_parameter("fps").value)

        self.frame_size = int(self.width * self.height * 3 // 2)

        self.declare_parameter("mm_per_px", MM_PER_PX)
        self.declare_parameter("auto_reset_invalid_frames", 15)
        self.declare_parameter("tof_topic", "/sensors/tof/distance_mm")
        self.declare_parameter("tof_reference_distance_mm", 114.5)
        self.declare_parameter("tof_alpha", 0.2)

        self.mm_per_px_base = float(self.get_parameter("mm_per_px").value)
        self._mm_lock = threading.Lock()
        self._mm_per_px = self.mm_per_px_base

        self.tof_topic = str(self.get_parameter("tof_topic").value)
        self.tof_reference_distance_mm = float(
            self.get_parameter("tof_reference_distance_mm").value
        )
        self.tof_alpha = float(self.get_parameter("tof_alpha").value)

        self.tof_sub = self.create_subscription(
            Float32,
            self.tof_topic,
            self._tof_callback,
            10,
        )

        self.tm_method = TM_METHOD
        self.patch_rel_offsets = PATCH_REL_OFFSETS

        self.image_pub = self.create_publisher(Image, "/rpi_camera/image_raw", 10)
        self.delta_px_pub = self.create_publisher(Vector3Stamped, "/vision/tm_delta_px", 10)
        self.pose_px_pub = self.create_publisher(Vector3Stamped, "/vision/tm_pose_px", 10)
        self.delta_mm_pub = self.create_publisher(Vector3Stamped, "/vision/tm_delta_mm", 10)
        self.pose_mm_pub = self.create_publisher(Vector3Stamped, "/vision/tm_pose_mm", 10)
        self.vel_mm_pub = self.create_publisher(Vector3Stamped, "/vision/tm_vel_mm", 10)
        self.quality_pub = self.create_publisher(Float32, "/vision/tm_quality", 10)
        self.valid_patches_pub = self.create_publisher(Int32, "/vision/tm_valid_patches", 10)

        self.initialized = False
        self.reset_requested = False

        self.W = None
        self.H = None
        self.patches = []
        self.baseline_center = None
        self.current_disp = np.zeros(2, dtype=np.float32)
        self.fusion_filter = ExponentialFusion(alpha=self.fusion_alpha)
        self.last_time = None
        self.frame_idx = 0
        self.consecutive_invalid_frames = 0

        self.declare_parameter("uart_port", "/dev/ttyAMA0")
        self.declare_parameter("uart_baud", 115200)
        uart_port = self.get_parameter("uart_port").value
        uart_baud = int(self.get_parameter("uart_baud").value)

        self.uart = None
        try:
            self.uart = serial.Serial(uart_port, uart_baud, timeout=0.01)
            self.get_logger().info(f"Opened UART on {uart_port} @ {uart_baud} baud")
        except Exception as e:
            self.get_logger().error(f"Failed to open UART: {e}")

        self.last_teensy_time = None
        self.last_teensy_pi_rx = None
        self._uart_rx_buf = b""

        self.cam_proc = self._start_camera_process()
        self._capture_running = True

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        if self.uart is not None:
            threading.Thread(target=self._uart_reader_loop, daemon=True).start()

    def _start_camera_process(self) -> subprocess.Popen:
        cmd = [
            "rpicam-vid",
            "--codec", "yuv420",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(int(self.fps)),
            "-t", "0",
            "-o", "-",
        ]

        self.get_logger().info("Starting camera process: " + " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            self.get_logger().error(
                "Camera binary not found. Adjust cmd in _start_camera_process()."
            )
            raise

        threading.Thread(
            target=self._log_camera_stderr,
            args=(proc,),
            daemon=True,
        ).start()

        return proc

    def _log_camera_stderr(self, proc: subprocess.Popen):
        for line in iter(proc.stderr.readline, b""):
            if not line:
                break
            txt = line.decode(errors="ignore").strip()
            if txt:
                self.get_logger().warn(f"[cam] {txt}")

    def _read_exact(self, n: int):
        buf = bytearray()
        stdout = self.cam_proc.stdout

        while len(buf) < n and self._capture_running:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)

        if not self._capture_running:
            return None

        return bytes(buf)

    def _capture_loop(self):
        self.get_logger().info("Capture loop started.")
        while self._capture_running and rclpy.ok():
            frame = self._read_exact(self.frame_size)
            if frame is None:
                self.get_logger().warning("Camera stdout ended or error in capture loop")
                break

            now = self.get_clock().now().to_msg()

            img_msg = Image()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "rpi_camera_optical_frame"
            img_msg.height = self.height
            img_msg.width = self.width
            img_msg.encoding = "yuv420"
            img_msg.is_bigendian = 0
            img_msg.step = self.width
            img_msg.data = frame
            self.image_pub.publish(img_msg)

            self._process_frame(now, frame)

        self.get_logger().info("Capture loop stopped")

    def _uart_reader_loop(self):
        self.get_logger().info("UART reader thread started (listening for Teensy timestamps).")
        while rclpy.ok() and self.uart is not None:
            try:
                chunk = self.uart.read(64)
                if not chunk:
                    continue

                self._uart_rx_buf += chunk

                while b"\n" in self._uart_rx_buf:
                    line, _, rest = self._uart_rx_buf.partition(b"\n")
                    self._uart_rx_buf = rest

                    line_str = line.decode("ascii", errors="ignore").strip()
                    if not line_str:
                        continue

                    if line_str.startswith("T,"):
                        try:
                            _, t_str = line_str.split(",", 1)
                            t_val = float(t_str)
                            self.last_teensy_time = t_val
                            self.last_teensy_pi_rx = time.time()
                        except ValueError:
                            continue
            except Exception as e:
                self.get_logger().warn(f"UART reader error: {e}")
                time.sleep(0.01)

    def _keyboard_loop(self):
        self.get_logger().info("Keyboard thread started. Press 'R' to reset TM patches.")
        try:
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch.lower() == "r":
                    self.get_logger().info("Template matching reset requested via keyboard.")
                    self.reset_requested = True
        except Exception as e:
            self.get_logger().warn(f"Keyboard loop error: {e}")

    def _initialize_patches(self, gray):
        self.H, self.W = gray.shape[:2]

        frame_center = np.array([self.W / 2.0, self.H / 2.0], dtype=np.float32)
        patches = []

        for idx, (rx, ry) in enumerate(self.patch_rel_offsets):
            cx = frame_center[0] + rx * self.W * 0.5
            cy = frame_center[1] + ry * self.H * 0.5
            template, tl, br, center = crop_template_around(
                gray, cx, cy, self.template_half, self.W, self.H
            )
            patches.append({
                "id": idx,
                "template": template,
                "size": template.shape[:2],
                "anchor_center": np.array(center, dtype=np.float32),
                "last_rect": (tl, br),
                "last_score": np.nan,
                "last_quality": np.nan,
            })

        if not patches:
            raise RuntimeError("No patches configured (this should not happen).")

        self.patches = patches
        self.baseline_center = np.mean([p["anchor_center"] for p in patches], axis=0)

        self.current_disp = np.zeros(2, dtype=np.float32)
        self.fusion_filter.reset()
        self.initialized = True
        self.frame_idx = 0
        self.last_time = None
        self.consecutive_invalid_frames = 0

        self.get_logger().info("Template patches initialized / re-initialized.")

    def _process_frame(self, stamp, frame_bytes: bytes):
        width = self.width
        height = self.height

        data = np.frombuffer(frame_bytes, dtype=np.uint8)
        y_size = width * height
        if data.size < y_size:
            self.get_logger().warn(
                f"Y plane size mismatch: data={data.size}, expected at least {y_size}"
            )
            return

        gray = data[:y_size].reshape((height, width))

        if self.last_teensy_time is not None and self.last_teensy_pi_rx is not None:
            t_pi_now = time.time()
            dt_pi = t_pi_now - self.last_teensy_pi_rx
            t_teensy_frame = self.last_teensy_time + dt_pi
        else:
            t_teensy_frame = None

        if (not self.initialized) or self.reset_requested:
            self._initialize_patches(gray)
            self.reset_requested = False
            self._publish_outputs(
                stamp=stamp,
                delta_disp_px=np.zeros(2, dtype=np.float32),
                quality=0.0,
                valid_count=len(self.patches),
                vx_mm=0.0,
                vy_mm=0.0,
                t_teensy_frame=t_teensy_frame,
            )
            return

        now_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(now_sec - self.last_time, 0.0)
        self.last_time = now_sec
        self.frame_idx += 1

        W, H = self.W, self.H
        current_disp = self.current_disp.copy()

        displacements = []
        qualities = []

        for patch in self.patches:
            template = patch["template"]
            ph, pw = patch["size"]
            pred_center = patch["anchor_center"] + current_disp

            top_left, score, quality = match_in_roi(
                gray, template, pred_center, self.search_half, W, H, self.tm_method
            )
            bottom_right = (top_left[0] + pw, top_left[1] + ph)
            center = np.array(
                (top_left[0] + pw / 2.0, top_left[1] + ph / 2.0),
                dtype=np.float32,
            )

            edge_hit = near_edge(top_left, pw, ph, W, H, self.pre_drop_margin_px)
            distance = float(np.linalg.norm(center - pred_center))
            distance_gate = distance <= (self.dist_gating_ratio * self.search_half)
            valid = (quality >= self.score_thresh) and not edge_hit and distance_gate

            patch["last_rect"] = (top_left, bottom_right)
            patch["last_score"] = score
            patch["last_quality"] = quality

            if valid:
                disp = center - patch["anchor_center"]
                displacements.append(disp)
                qualities.append(quality)

        raw_disp = None
        valid_count = len(displacements)
        aggregated_quality = float("nan")

        if valid_count >= self.min_valid_patches:
            disp_arr = np.stack(displacements, axis=0)
            qual_arr = np.array(qualities, dtype=np.float32)

            median_disp = np.median(disp_arr, axis=0)
            residuals = np.linalg.norm(disp_arr - median_disp, axis=1)
            keep_mask = residuals <= self.residual_gating_px

            if keep_mask.sum() >= self.min_valid_patches:
                disp_arr = disp_arr[keep_mask]
                qual_arr = qual_arr[keep_mask]

            weights_arr = np.maximum(qual_arr, 1e-6) ** self.quality_power
            weights_sum = float(weights_arr.sum())
            if weights_sum > 0.0:
                weights_norm = weights_arr / weights_sum
                raw_disp = (weights_norm[:, None] * disp_arr).sum(axis=0)
                aggregated_quality = float((weights_norm * qual_arr).sum())
            else:
                raw_disp = median_disp
                aggregated_quality = float("nan")

            self.consecutive_invalid_frames = 0
        else:
            aggregated_quality = 0.0
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames >= self.auto_reset_invalid_frames:
                self.get_logger().warn(
                    f"No valid TM patches for {self.consecutive_invalid_frames} frames. Auto-resetting."
                )
                self.reset_requested = True

        fused_disp = self.fusion_filter.update(raw_disp)
        if fused_disp is not None:
            new_disp = fused_disp
        else:
            new_disp = current_disp

        delta_disp = new_disp - current_disp
        self.current_disp = new_disp.astype(np.float32)

        dx_px = float(delta_disp[0])
        dy_px = float(delta_disp[1])
        x_px = float(self.current_disp[0])
        y_px = float(self.current_disp[1])

        dx_mm = dx_px * self.mm_per_px
        dy_mm = dyPx * self.mm_per_px

        if dt > 1e-9:
            vx_mm = dx_mm / dt
            vy_mm = dy_mm / dt
        else:
            vx_mm = 0.0
            vy_mm = 0.0

        self.get_logger().debug(
            f"TM frame {self.frame_idx} | "
            f"Δx={dx_px:.3f}px Δy={dy_px:.3f}px | "
            f"x={x_px:.3f}px y={y_px:.3f}px | "
            f"quality={aggregated_quality:.3f} | "
            f"valid_patches={valid_count} | dt={dt:.4f}s"
        )

        self._publish_outputs(
            stamp=stamp,
            delta_disp_px=delta_disp,
            quality=aggregated_quality,
            valid_count=valid_count,
            vx_mm=vx_mm,
            vy_mm=vy_mm,
            t_teensy_frame=t_teensy_frame,
        )

    def _publish_outputs(
        self,
        stamp,
        delta_disp_px: np.ndarray,
        quality: float,
        valid_count: int,
        vx_mm: float = 0.0,
        vy_mm: float = 0.0,
        t_teensy_frame: float = None,
    ):
        dx_px = float(delta_disp_px[0])
        dy_px = float(delta_disp_px[1])
        x_px = float(self.current_disp[0])
        y_px = float(self.current_disp[1])

        dx_mm = dx_px * self.mm_per_px
        dy_mm = dy_px * self.mm_per_px
        x_mm = x_px * self.mm_per_px
        y_mm = y_px * self.mm_per_px

        msg_d_px = Vector3Stamped()
        msg_d_px.header.stamp = stamp
        msg_d_px.header.frame_id = "rpi_camera_optical_frame"
        msg_d_px.vector.x = dx_px
        msg_d_px.vector.y = dy_px
        msg_d_px.vector.z = 0.0
        self.delta_px_pub.publish(msg_d_px)

        msg_p_px = Vector3Stamped()
        msg_p_px.header.stamp = stamp
        msg_p_px.header.frame_id = "rpi_camera_optical_frame"
        msg_p_px.vector.x = x_px
        msg_p_px.vector.y = y_px
        msg_p_px.vector.z = 0.0
        self.pose_px_pub.publish(msg_p_px)

        msg_d_mm = Vector3Stamped()
        msg_d_mm.header.stamp = stamp
        msg_d_mm.header.frame_id = "rpi_camera_optical_frame"
        msg_d_mm.vector.x = dx_mm
        msg_d_mm.vector.y = dy_mm
        msg_d_mm.vector.z = 0.0
        self.delta_mm_pub.publish(msg_d_mm)

        msg_p_mm = Vector3Stamped()
        msg_p_mm.header.stamp = stamp
        msg_p_mm.header.frame_id = "rpi_camera_optical_frame"
        msg_p_mm.vector.x = x_mm
        msg_p_mm.vector.y = y_mm
        msg_p_mm.vector.z = 0.0
        self.pose_mm_pub.publish(msg_p_mm)

        msg_v_mm = Vector3Stamped()
        msg_v_mm.header.stamp = stamp
        msg_v_mm.header.frame_id = "rpi_camera_optical_frame"
        msg_v_mm.vector.x = vx_mm
        msg_v_mm.vector.y = vy_mm
        msg_v_mm.vector.z = 0.0
        self.vel_mm_pub.publish(msg_v_mm)

        q_msg = Float32()
        q_msg.data = float(quality if np.isfinite(quality) else 0.0)
        self.quality_pub.publish(q_msg)

        n_msg = Int32()
        n_msg.data = int(valid_count)
        self.valid_patches_pub.publish(n_msg)

        if getattr(self, "uart", None) is not None and self.uart.writable():
            if t_teensy_frame is None:
                return
            try:
                line = (
                    f"{t_teensy_frame:.6f},"
                    f"{x_mm:.3f},{y_mm:.3f},"
                    f"{dx_mm:.3f},{dy_mm:.3f},"
                    f"{vx_mm:.3f},{vy_mm:.3f}\n"
                )
                self.uart.write(line.encode("ascii"))
            except Exception as e:
                self.get_logger().warn(f"UART write failed: {e}")

    def destroy_node(self):
        self._capture_running = False
        try:
            if self.cam_proc is not None:
                self.get_logger().info("Terminating camera process...")
                self.cam_proc.terminate()
        except Exception:
            pass
        try:
            if self.uart is not None:
                self.uart.close()
        except Exception:
            pass
        super().destroy_node()

    def _tof_callback(self, msg: Float32):
        distance_mm = float(msg.data)
        if not math.isfinite(distance_mm) or distance_mm <= 0.0:
            return

        ref = self.tof_reference_distance_mm if self.tof_reference_distance_mm > 0.0 else distance_mm
        new_mm_per_px = max(1e-6, self.mm_per_px_base * (distance_mm / ref))

        with self._mm_lock:
            if 0.0 < self.tof_alpha < 1.0:
                self._mm_per_px = (1.0 - self.tof_alpha) * self._mm_per_px + self.tof_alpha * new_mm_per_px
            else:
                self._mm_per_px = new_mm_per_px

    def _get_mm_per_px(self) -> float:
        with self._mm_lock:
            return float(self._mm_per_px)


def main(args=None):
    rclpy.init(args=args)
    node = TemplateMatchingCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()