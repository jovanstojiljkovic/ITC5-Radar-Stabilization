import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from vl53l5cx.vl53l5cx import VL53L5CX
from vl53l5cx.api import VL53L5CX_RESOLUTION_8X8


class ToFDistancePublisher(Node):
    def __init__(self):
        super().__init__("tof_distance_publisher")

        self.declare_parameter("publish_topic", "/sensors/tof/distance_mm")
        self.declare_parameter("publish_mode", "median")
        self.declare_parameter("sensor_frequency_hz", 15.0)

        topic = str(self.get_parameter("publish_topic").value)
        self.publish_mode = str(self.get_parameter("publish_mode").value).lower()
        self.frequency = float(self.get_parameter("sensor_frequency_hz").value)

        self.publisher = self.create_publisher(Float32, topic, 10)

        self.driver = VL53L5CX()
        if not self.driver.is_alive():
            raise RuntimeError("VL53L5CX device is not alive")

        self.driver.init()
        self.driver.set_resolution(VL53L5CX_RESOLUTION_8X8)
        self.driver.set_ranging_frequency_hz(int(self.frequency))

        resolution = self.driver.get_resolution()
        self._side = int(math.isqrt(resolution))

        self.driver.start_ranging()

        period = 1.0 / max(self.frequency, 1.0)
        self.timer = self.create_timer(period, self._timer_cb)

    def _timer_cb(self):
        if not self.driver.check_data_ready():
            return

        data = self.driver.get_ranging_data()
        distances = np.array(data.distance_mm, dtype=np.float32).reshape((self._side, self._side))

        if self.publish_mode == "center":
            idx = self._side // 2
            measurement = float(distances[idx, idx])
        elif self.publish_mode == "mean":
            measurement = float(np.mean(distances))
        else:
            measurement = float(np.median(distances))

        if not math.isfinite(measurement) or measurement <= 0.0:
            return

        msg = Float32()
        msg.data = measurement
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.driver.stop_ranging()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ToFDistancePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()