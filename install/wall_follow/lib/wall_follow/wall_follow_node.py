#!/usr/bin/env python3
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

        # create subscribers and publishers
        self.subscriber_ =  self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # set PID gains
        self.kp = 1.5    #0.25          0.4~0.6 can turn the corner
        self.kd = 0.05     #0.5
        self.ki = 0.25

        # store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0

        # store any necessary values you think you'll need
        self.desired_distance = 1.0
        self.lookahead_distance = 0.9
        self.dt = 0.0  # 3ms
        self.prev_time = 0.0

        self.prev_D_t1_left = 0.0
        self.d_filtered = 0.0
        self.deriv_tau = 0.15  

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

        # ------------------- LEFT WALL -------------------
        angle_b_left = np.deg2rad(90)
        angle_a_left = angle_b_left - theta

        a_left = self.get_range(range_data, angle_a_left)
        b_left = self.get_range(range_data, angle_b_left)
        

        # self.get_logger().info(f"a_left={a_left:+.3f} | b_left={b_left:+.3f}")

        alpha_left = np.arctan2(abs(a_left * np.cos(theta) - b_left), a_left * np.sin(theta))
        D_t_left = b_left * np.cos(alpha_left)

        # self.get_logger().info(f"D_t_left={D_t_left:+.3f} | alpha_left={alpha_left:+.3f}")

        D_t1_left = D_t_left + self.lookahead_distance * np.sin(alpha_left)

        # Smooth the sudden change rate
        # if self.prev_D_t1_left == 0.0:
        #     self.prev_D_t1_left = D_t1_left
        
        # D_t1_left_delta = abs(D_t1_left - self.prev_D_t1_left)

        # if D_t1_left_delta > 0.35:
        #     D_t1_left = self.prev_D_t1_left + 0.2 * (D_t1_left - self.prev_D_t1_left)
        #     self.prev_D_t1_left = D_t1_left
        # else:
        #     self.prev_D_t1_left = D_t1_left
        
        # self.get_logger().info(f"D_t1_left={D_t1_left:+.3f}")

        # ------------------- RIGHT WALL -------------------
        # angle_b_right = np.deg2rad(-90)
        # angle_a_right = angle_b_right + theta

        # a_right = self.get_range(range_data, angle_a_right)
        # b_right = self.get_range(range_data, angle_b_right)

        # # self.get_logger().info(f"a_right={a_right:+.3f} | b_right={b_right:+.3f}")

        # alpha_right = np.arctan2(abs(a_right * np.cos(theta) - b_right), a_right * np.sin(theta))
        # D_t_right = b_right * np.cos(alpha_right)

        # # self.get_logger().info(f"D_t_right={D_t_right:+.3f} | alpha_right={alpha_right:+.3f}")

        # D_t1_right = D_t_right + self.lookahead_distance * np.sin(alpha_right)


        # ------------------- Center error ------------------
        # if(D_t1_left + D_t1_right > 1.9 and abs(D_t1_left - D_t1_right) < 0.3):
        #     error = D_t1_right - D_t1_left
        # else:
        #     error = dist - D_t1_left


        # Nasty Trick
        # self.get_logger().info(f"left + right={D_t1_left + D_t1_right:+.3f}")

        # if  3.7 > (D_t1_left + D_t1_right) > 2:
        #     D_t1_left = 1.1

        error = dist - D_t1_left


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

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        # self.prev_error = error

        # Low pass filter. Try to get through the ditch
        # lp_filter = self.deriv_tau / (self.deriv_tau + self.dt)
        # self.d_filtered = lp_filter * self.d_filtered + (1 - lp_filter) * derivative

        self.prev_error = error

        angle = -(self.kp * error + self.ki * self.integral + self.kd * derivative)

        # steer angle is between -30~30 degree
        # angle = float(np.clip(angle, -np.deg2rad(30), np.deg2rad(30)))
        
        abs_angle_deg = abs(np.rad2deg(angle))

        if abs_angle_deg < 10:
            speed = 1.8
        elif abs_angle_deg < 20:
            speed = 1.2
        else:
            speed = 0.8

        speed = min(speed, velocity)

        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed
        self.drive_publisher_.publish(drive_msg)

        # self.get_logger().info(f"err={error:+.3f} | steer={np.rad2deg(angle):+.2f}° | speed={speed:.2f}")

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        # self.get_logger().info(f"Start: Scan_callback")

        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

        # Calculate time interval
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_time == 0.0:
            self.dt = 0.03
        else:
            self.dt = curr_time - self.prev_time

        self.prev_time = curr_time

        # Get range data
        range_data = np.array(msg.ranges)
        # self.get_logger().info(f"Scan_callback: Get range")

        error = self.get_error(range_data, self.desired_distance) 
        # self.get_logger().info(f"Scan_callback: Get error = {error:+.3f}")

        velocity = 3.0 
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