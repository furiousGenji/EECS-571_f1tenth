import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.subscriber_ =  self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # TODO: set PID gains
        self.kp = 1.0
        self.kd = 0.0
        self.ki = 0.1

        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # TODO: store any necessary values you think you'll need

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        angle_min = self.angle_min
        angle_increment = self.angle_increment

        idx = int((angle - angle_min) / angle_increment)
        val = range_data[idx]
        if np.isfinite(val):
            return val
        
        return 0.0

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        theta = np.deg2rad(45)
        angle_b = np.deg2rad(90)
        angle_a = angle_b - theta

        a = self.get_range(range_data, angle_a)
        b = self.get_range(range_data, angle_b)

        alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        D_t = b * np.cos(alpha)
        D_t1 = D_t + self.lookahead_distance * np.sin(alpha)

        error = dist - D_t1
        return error

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        angle = self.kp * error + self.ki * self.integral + self.kd * derivative
        # angle = float(np.clip(angle, np.deg2rad(-30), np.deg2rad(30)))

        abs_angle_deg = abs(np.rad2deg(angle))

        if abs_angle_deg < 10:
            speed = 1.5
        elif abs_angle_deg < 20:
            speed = 1.0
        else:
            speed = 0.5

        speed = min(speed, velocity)

        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.drive_publisher_.publish(drive_msg)

        self.get_logger().info(f"err={error:+.3f} | steer={np.rad2deg(angle):+.2f}° | speed={speed:.2f}")

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

        range_data = np.array(msg.ranges)

        error = self.get_error(range_data, self.desired_distance) 

        velocity = 1.5 # TODO: calculate desired car velocity based on error
        self.pid_control(error, velocity)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()