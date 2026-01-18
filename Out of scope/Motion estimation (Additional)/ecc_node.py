# ECC camera tracker node using rpicam-vid with YUV420 output
# Utilized for comparison of motion estimation methods
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


class ECCLiveNode(Node):
    def __init__(self):
        super().__init__("camera_ecc_node")

    # Camera parameters
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 80)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = float(self.get_parameter("fps").value)

        # YUV420p: 1.5 bytes per pixel
        self.frame_size = int(self.width * self.height * 3 // 2)

    # ECC parameters
        self.declare_parameter("mm_per_px", 0.157566)
        self.declare_parameter("ecc_scale", 0.6)
        self.declare_parameter("patch_frac_single", 0.30)
        self.declare_parameter("ecc_max_iter", 50)
        self.declare_parameter("ecc_eps", 1e-2)
        self.declare_parameter("ecc_min_cc", 0.4)
        self.declare_parameter("auto_reset_invalid_frames", 15)
        self.declare_parameter("tof_topic", "/sensors/tof/distance_mm")
        self.declare_parameter("tof_reference_distance_mm", 114.5)
        self.declare_parameter("tof_alpha", 0.2)

        self.mm_per_px_base = float(self.get_parameter("mm_per_px").value)
        self._mm_lock = threading.Lock()
        self._mm_per_px = self.mm_per_px_base

        self.ecc_scale = float(self.get_parameter("ecc_scale").value)
        self.patch_frac_single = float(self.get_parameter("patch_frac_single").value)
        self.ecc_max_iter = int(self.get_parameter("ecc_max_iter").value)
        self.ecc_eps = float(self.get_parameter("ecc_eps").value)
        self.ecc_min_cc = float(self.get_parameter("ecc_min_cc").value)
        self.auto_reset_invalid_frames = int(
            self.get_parameter("auto_reset_invalid_frames").value
        )

        self.criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_max_iter,
            self.ecc_eps,
        )

    # Publishers
        self.image_pub = self.create_publisher(Image, "/rpi_camera/image_raw", 10)

        self.delta_px_pub = self.create_publisher(Vector3Stamped, "/vision/ecc_delta_px", 10)
        self.pose_px_pub = self.create_publisher(Vector3Stamped, "/vision/ecc_pose_px", 10)
        self.delta_mm_pub = self.create_publisher(Vector3Stamped, "/vision/ecc_delta_mm", 10)
        self.pose_mm_pub = self.create_publisher(Vector3Stamped, "/vision/ecc_pose_mm", 10)
        self.vel_mm_pub = self.create_publisher(Vector3Stamped, "/vision/ecc_vel_mm", 10)
        self.quality_pub = self.create_publisher(Float32, "/vision/ecc_quality", 10)
        self.valid_pub = self.create_publisher(Int32, "/vision/ecc_valid", 10)

    # ECC internal state
        self.initialized = False
        self.reset_requested = False

        self.W_full = None
        self.H_full = None
        self.w_s = None
        self.h_s = None

        self.prev_gray_scaled = None
        self.patch_scaled = None
        self.warp = np.eye(2, 3, dtype=np.float32)

        self.cumulative = np.zeros(2, dtype=np.float32)
        self.frame_idx = 0
        self.consecutive_invalid_frames = 0
        self.last_time = None

    # UART handling
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

    # Camera process
        self.cam_proc = self._start_camera_process()
        self._capture_running = True

    # Threads
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        if self.uart is not None:
            threading.Thread(target=self._uart_reader_loop, daemon=True).start()

    # Camera process helpers
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

    # UART reader
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

    # Keyboard reset
    def _keyboard_loop(self):
        self.get_logger().info("Keyboard thread started. Press 'R' to reset ECC.")
        try:
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch.lower() == "r":
                    self.get_logger().info("ECC reset requested via keyboard.")
                    self.reset_requested = True
        except Exception as e:
            self.get_logger().warn(f"Keyboard loop error: {e}")

    # ECC helpers
    def _make_single_patch_scaled(self, w_s, h_s):
        pw = int(max(1, self.patch_frac_single * w_s))
        ph = int(max(1, self.patch_frac_single * h_s))
        cx = w_s // 2
        cy = h_s // 2

        x0 = max(0, cx - pw // 2)
        y0 = max(0, cy - ph // 2)
        x1 = min(w_s, cx + pw // 2)
        y1 = min(h_s, cy + ph // 2)
        return (x0, y0, x1, y1)

    def _initialize_from_frame(self, gray_full):
        self.H_full, self.W_full = gray_full.shape[:2]

        self.w_s = int(self.W_full * self.ecc_scale)
        self.h_s = int(self.H_full * self.ecc_scale)
        if self.w_s < 10 or self.h_s < 10:
            raise RuntimeError("Scaled image too small for ECC.")

        gray_scaled = cv2.resize(
            gray_full,
            (self.w_s, self.h_s),
            interpolation=cv2.INTER_AREA,
        )

        self.patch_scaled = self._make_single_patch_scaled(self.w_s, self.h_s)
        self.prev_gray_scaled = gray_scaled
        self.warp = np.eye(2, 3, dtype=np.float32)

        self.cumulative = np.zeros(2, dtype=np.float32)
        self.frame_idx = 0
        self.consecutive_invalid_frames = 0
        self.last_time = None

        self.initialized = True
        self.get_logger().info(
            f"ECC initialized: full={self.W_full}x{self.H_full}, "
            f"scaled={self.w_s}x{self.h_s}, patch={self.patch_scaled}"
        )

    # Main frame processing
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

        gray_full = data[:y_size].reshape((height, width))

        now_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        if self.last_teensy_time is not None and self.last_teensy_pi_rx is not None:
            t_pi_now = time.time()
            dt_pi = t_pi_now - self.last_teensy_pi_rx
            t_teensy_frame = self.last_teensy_time + dt_pi
        else:
            t_teensy_frame = None

        if (not self.initialized) or self.reset_requested:
            self._initialize_from_frame(gray_full)
            self.reset_requested = False
            self._publish_outputs(
                stamp=stamp,
                step_dx=0.0,
                step_dy=0.0,
                quality=0.0,
                valid=0,
                dt=0.0,
                t_teensy_frame=t_teensy_frame,
            )
            return

        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(now_sec - self.last_time, 0.0)
        self.last_time = now_sec

        gray_scaled = cv2.resize(
            gray_full,
            (self.w_s, self.h_s),
            interpolation=cv2.INTER_AREA,
        )

        self.frame_idx += 1

        if self.prev_gray_scaled is None:
            self.prev_gray_scaled = gray_scaled
            self._publish_outputs(
                stamp=stamp,
                step_dx=0.0,
                step_dy=0.0,
                quality=0.0,
                valid=0,
                dt=dt,
                t_teensy_frame=t_teensy_frame,
            )
            return

        (x0, y0, x1, y1) = self.patch_scaled
        prev_roi = self.prev_gray_scaled[y0:y1, x0:x1]
        curr_roi = gray_scaled[y0:y1, x0:x1]

        step_dx_scaled = 0.0
        step_dy_scaled = 0.0
        quality = 0.0
        valid = 0

        try:
            warp_guess = self.warp.copy()
            cc, warp_res = cv2.findTransformECC(
                prev_roi,
                curr_roi,
                warp_guess,
                cv2.MOTION_TRANSLATION,
                self.criteria,
                None,
                1,
            )
            self.warp = warp_res
            quality = float(cc)

            if np.isfinite(quality) and quality >= self.ecc_min_cc:
                step_dx_scaled = float(warp_res[0, 2])
                step_dy_scaled = float(warp_res[1, 2])
                valid = 1
            else:
                self.warp = np.eye(2, 3, dtype=np.float32)

        except cv2.error:
            self.warp = np.eye(2, 3, dtype=np.float32)

        step_dx = step_dx_scaled / self.ecc_scale
        step_dy = step_dy_scaled / self.ecc_scale

        if valid:
            self.cumulative += np.array([step_dx, step_dy], dtype=np.float32)
            self.consecutive_invalid_frames = 0
        else:
            step_dx = 0.0
            step_dy = 0.0
            quality = 0.0
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames >= self.auto_reset_invalid_frames:
                self.get_logger().warn(
                    f"ECC invalid for {self.consecutive_invalid_frames} frames. Auto-resetting."
                )
                self.reset_requested = True

        self.prev_gray_scaled = gray_scaled

        self.get_logger().debug(
            f"ECC frame {self.frame_idx} | "
            f"Δx={step_dx:.3f}px Δy={step_dy:.3f}px | "
            f"x={self.cumulative[0]:.3f}px y={self.cumulative[1]:.3f}px | "
            f"quality={quality:.3f} valid={valid}"
        )

        self._publish_outputs(
            stamp=stamp,
            step_dx=step_dx,
            step_dy=step_dy,
            quality=quality,
            valid=valid,
            dt=dt,
            t_teensy_frame=t_teensy_frame,
        )

    # Publishing helper
    def _publish_outputs(self, stamp, step_dx, step_dy, quality, valid, dt, t_teensy_frame):
        x_px = float(self.cumulative[0])
        y_px = float(self.cumulative[1])

        dx_px = float(step_dx)
        dy_px = float(step_dy)

        mm_per_px = self._get_mm_per_px()
        dx_mm = dx_px * mm_per_px
        dy_mm = dy_px * mm_per_px
        x_mm = x_px * mm_per_px
        y_mm = y_px * mm_per_px

        if dt > 0.0:
            vx_mm = dx_mm / dt
            vy_mm = dy_mm / dt
        else:
            vx_mm = 0.0
            vy_mm = 0.0

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

        v_msg = Int32()
        v_msg.data = int(valid)
        self.valid_pub.publish(v_msg)

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

    # TOF callback
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

    # Cleanup
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


def main(args=None):
    rclpy.init(args=args)
    node = ECCLiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()