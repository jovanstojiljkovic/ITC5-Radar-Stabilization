# Depracated camera node using rpicam-vid with YUV420 output
# Publishes raw YUV420 frames over ROS2 topic /rpi_camera/image_raw
# Combined with pyr_lk_node.py for motion estimation
import threading
import subprocess

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RPICameraNode(Node):
    def __init__(self):
        super().__init__('rpi_camera_node')

        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 80)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)

        # YUV420p: 1.5 bytes per pixel
        self.frame_size = int(self.width * self.height * 3 // 2)

        # Start the camera process (THIS is where --codec yuv420 goes)
        self.cam_proc = self._start_camera_process()

        self.latest_frame = None
        self._capture_running = True

        # Background capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.capture_thread.start()

        # ROS publisher & timer
        self.pub = self.create_publisher(Image, '/rpi_camera/image_raw', 10)
        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f"RPICameraNode started: {self.width}x{self.height} @ {self.fps} fps "
            f"(frame_size={self.frame_size} bytes, encoding='yuv420')"
        )

    # Raw image fork
    def _start_camera_process(self) -> subprocess.Popen:
        
        cmd = [
            'rpicam-vid',              
            '--codec', 'yuv420',
            '--width', str(self.width),
            '--height', str(self.height),
            '--framerate', str(int(self.fps)),
            '-t', '0',                  # run indefinitely
            '-o', '-'                   # output to stdout
        ]

        self.get_logger().info("Starting camera process: " + " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # unbuffered
            )
        except FileNotFoundError:
            self.get_logger().error(
                "Camera binary not found. Adjust cmd in _start_camera_process()."
            )
            raise

        # log stderr
        threading.Thread(
            target=self._log_camera_stderr,
            args=(proc,),
            daemon=True,
        ).start()

        return proc

    def _log_camera_stderr(self, proc: subprocess.Popen):
        """Forward camera stderr lines into ROS logs."""
        for line in iter(proc.stderr.readline, b''):
            if not line:
                break
            txt = line.decode(errors='ignore').strip()
            if txt:
                self.get_logger().warn(f"[cam] {txt}")

    # Raw camera reader
    def _read_exact(self, n: int):
      
        buf = bytearray()
        stdout = self.cam_proc.stdout

        while len(buf) < n and self._capture_running:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                # EOF or error
                return None
            buf.extend(chunk)

        if not self._capture_running:
            return None

        return bytes(buf)

    def _capture_loop(self):
        
        while self._capture_running:
            frame = self._read_exact(self.frame_size)
            if frame is None:
                self.get_logger().warning(
                    "Camera stdout ended or error in capture loop"
                )
                break

            self.latest_frame = frame

        self.get_logger().info("Capture loop stopped")

    # Timer callback to publish latest frame
    def timer_callback(self):
        data = self.latest_frame
        if data is None:
            return

        if len(data) != self.frame_size:
            self.get_logger().warning(
                f"Latest frame size mismatch: {len(data)} vs expected {self.frame_size}"
            )
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'rpi_camera_optical_frame'
        msg.height = self.height
        msg.width = self.width
        msg.encoding = 'yuv420'   
        msg.is_bigendian = 0
        msg.step = self.width  
        msg.data = data

        self.pub.publish(msg)

    # cleanup helper
    def destroy_node(self):
        self._capture_running = False
        try:
            if self.cam_proc is not None:
                self.get_logger().info("Terminating camera process...")
                self.cam_proc.terminate()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RPICameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
