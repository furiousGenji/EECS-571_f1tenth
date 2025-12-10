import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

import time


class ArbiterNode(Node):
    def __init__(self):
        super().__init__("arbiter_node")

        self.total_compute_time = 0
        self.compute_count = 0
        self.arbite_count = 0
        self.MLP_count = 0
        self.FTG_count = 0

        # Parameters
        self.declare_parameter("mlp_threshold", 0.70)   # default threshold
        self.threshold = self.get_parameter("mlp_threshold").value

        # Storage for latest messages
        self.ftg_msg = None
        self.mlp_msg = None
        self.mlp_conf = 0.0

        # Subscriptions
        self.create_subscription(
            AckermannDriveStamped, "/drive", self.ftg_callback, 10)

        self.create_subscription(
            AckermannDriveStamped, "/drive_mlp", self.mlp_callback, 10)

        self.create_subscription(
            Float32, "/mlp_confidence", self.conf_callback, 10)

        # Publisher for final command
        self.final_pub = self.create_publisher(
            AckermannDriveStamped, "/drive_final", 10)

        # self.get_logger().info("ArbiterNode initialized!")

    def ftg_callback(self, msg):

        # self.get_logger().info("FTG RECEIVED")
        self.ftg_msg = msg
        self.make_decision()
    def mlp_callback(self, msg):

        # self.get_logger().info("MLP RECEIVED")
        self.mlp_msg = msg
        self.make_decision()

    def conf_callback(self, msg):
        # self.get_logger().info(f"CONF RECEIVED: {msg.data:.3f}")
        self.mlp_conf = msg.data

    def make_decision(self):

        # t0 = time.perf_counter()

        # Ensure both messages arrived
        if self.ftg_msg is None or self.mlp_msg is None:
            return

        # Read threshold
        threshold = self.get_parameter("mlp_threshold").value

        # Choose which controller to trust
        if self.mlp_conf > threshold:
           chosen = self.mlp_msg
           source = "MLP"
           self.MLP_count += 1
        else:
            chosen = self.ftg_msg
            source = "FTG"
            self.FTG_count += 1

        # Extract chosen steering/speed
        steer = chosen.drive.steering_angle
        speed = chosen.drive.speed

        # Detailed Logging
        # self.get_logger().info(
        #    f"[ARB] → {source} chosen | "
        #    f"steer={steer:.3f}, speed={speed:.3f}, "
        #    f"conf={self.mlp_conf:.3f}, thresh={threshold:.2f}"
        #  )

        # Publish final decision
        self.final_pub.publish(chosen)

        # t1 = time.perf_counter()
        # dt = t1 - t0 

        # self.total_compute_time += dt
        # self.compute_count += 1

        # if self.compute_count % 5000 == 0:
        #     avg_ms = (self.total_compute_time / self.compute_count) * 1000.0
        #     self.get_logger().info(f"Average compute time = {avg_ms:.3f} ms/step")

        self.arbite_count += 1

        if self.arbite_count % 5000 == 0:
            avg_MLP = self.MLP_count / self.arbite_count
            self.get_logger().info(f"Average MLP time = {avg_MLP:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = ArbiterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
