#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        self.prev_ranges = None
        self.prev_time = None

        # create ROS subscribers and publishers.
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.scan_subscriber_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscriber_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        

    def odom_callback(self, odom_msg):
        # update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # calculate TTC
        # print(">>> Odom callback triggered")
        # Record current time
        curr_time = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9

        # Record current range
        curr_ranges = np.array(scan_msg.ranges)

        # Initialization
        if self.prev_ranges is None:
            self.prev_ranges = curr_ranges
            self.prev_time = curr_time
            self.get_logger().info("Initialization")
            return

        # Time variation
        dt = curr_time - self.prev_time

        # period 30ms
        if(dt <= 0.03):
            return
        

        # Select data from front angle -30 ~ 30 degree #### Change HERE to manipulate the angle #####
        idx_min = int((-np.deg2rad(30) - scan_msg.angle_min) / scan_msg.angle_increment)
        idx_max = int((np.deg2rad(30) - scan_msg.angle_min) / scan_msg.angle_increment)

        curr_ranges = curr_ranges[idx_min : idx_max + 1]
        prev_ranges = self.prev_ranges[idx_min : idx_max + 1]

        # Eliminate inf/NaN
        valid_mask = np.isfinite(curr_ranges) & np.isfinite(prev_ranges)
        if not np.any(valid_mask):
            self.get_logger().info("inf/NaN")
            return
        
        curr_ranges = curr_ranges[valid_mask]
        prev_ranges = prev_ranges[valid_mask]

        range_dot = (curr_ranges - prev_ranges) / dt

        range_dot_rst = []
        for i in range_dot:
            if -i > 0:
                range_dot_rst.append(-i)
            else:
                range_dot_rst.append(1e-6)
        
        range_dot_rst = np.array(range_dot_rst)

        iTTC = curr_ranges / range_dot_rst

        # Eliminate inf/NaN
        iTTC = iTTC[np.isfinite(iTTC)]
        if iTTC.size == 0:
            self.get_logger().info("No valid iTTC values this frame")
            return

        # iTTC_min = np.min(iTTC)
        iTTC_min = np.percentile(iTTC, 5)
        self.get_logger().info(f"iTTC min = {iTTC_min:.2f}")

        self.prev_ranges = np.array(scan_msg.ranges)
        self.prev_time = curr_time

        #Set iTTC threshold ##### Change HERE to manipulate the TTC #####
        if(iTTC_min > 0.7 or np.isinf(iTTC_min)):
            return
        else:
            set_speed = 0.0
        
            # publish command to brake
            new_scan_msg = AckermannDriveStamped()
            new_scan_msg.drive.speed = set_speed
            self.drive_publisher_.publish(new_scan_msg)

            rclpy.shutdown()
        
        pass

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()