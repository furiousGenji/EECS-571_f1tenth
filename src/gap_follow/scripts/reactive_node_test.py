#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.scan_subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # Publish to drive
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.max_range_clip = 8.0
        self.movavg_window = 5
        self.bubble_points = 40 #40 

        self.speed_lo = 0.8
        self.speed_md = 1.2
        self.speed_hi = 1.8
        self.small_turn_deg = 10.0
        self.mid_turn_deg = 20.0
        self.max_steer = math.radians(90)
        self.fov = 180.0
        self.fov_min_id = 0.0

        self.disp_thresh = 0.35
        self.car_half_width = 0.12
    
    def apply_disparity_extender(self, free_space, angle_increment):

        free_space_origin = free_space.copy()
        # mask = np.zeros_like(free_space_origin, dtype=bool)

        diff = np.diff(free_space)
        
        disp_idx = np.where(np.abs(diff) > self.disp_thresh)[0]
        if disp_idx.size == 0:
            return

        n = free_space.size
        for i in disp_idx:
            d1, d2 = free_space_origin[i], free_space_origin[i + 1]

            # self.get_logger().info(f"disparity d1[{i}] = {d1}")
            # self.get_logger().info(f"disparity d2[{i}] = {d2}")
            
            d_near = min(d1,d2)
            safety_factor = 1.0 + np.exp(-d_near / 1.5)

            # self.get_logger().info(f"disparity d_near[{i}] = {d_near}")

            theta = math.atan2(self.car_half_width * safety_factor, float(d_near))

            # self.get_logger().info(f"disparity theta[{i}] = {theta}")
            
            points = int(math.ceil(theta / angle_increment))

            # self.get_logger().info(f"disparity points[{i}] = {points}")

            atten = np.linspace(1.0, 0.2, points)

            if d1 < d2:
                right = min(n, i + 1 + points)
                free_space[i + 1 : right] *= atten[:right - (i+1)]
            else:
                left = max(0, i + 1 - points)
                free_space[left : i + 1] *= atten[-(i+1 - left):]



    def preprocess_lidar(self, ranges, angle_min, angle_max, angle_increment):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        arr = np.array(ranges)

        # i=0
        # for val in arr:
        #     self.get_logger().info(f"arr[{i}]={val}")
        #     i += 1

        # truncate vision range from -fov ~fov
        fov_rad = math.radians(self.fov)
        fov_min_id = int (((-fov_rad / 2) - angle_min) / angle_increment)
        fov_max_id = int (((fov_rad / 2) - angle_min) / angle_increment)
        self.fov_min_id = fov_min_id

        arr = arr[fov_min_id:fov_max_id]
        # self.get_logger().info(f"fov_min_id={fov_min_id}, fov_max_id={fov_max_id}, len={len(ranges)}")

        #Rejecting high values
        np.clip(arr, 0.0, self.max_range_clip, out=arr)

        # i=0
        # for val in arr:
        #     self.get_logger().info(f"cliped_arr[{i}]={val}")
        #     i += 1

        if self.movavg_window > 1:
            k = self.movavg_window
            
            kernel = np.ones(k, dtype=np.float32) / float(k)
            
            pad_left = np.repeat(arr[0], (k - 1) // 2)
            pad_right = np.repeat(arr[-1], (k - 1) // 2)
            padded = np.concatenate([pad_left, arr, pad_right])
            arr = np.convolve(padded, kernel, mode='valid')

        # i=0
        # for val in arr:
        #     self.get_logger().info(f"slidingWindow_arr[{i}]={val}")
        #     i += 1

        return arr


    def find_best_point(self, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """

        maxval = np.max(ranges)
        mask = np.equal(ranges, maxval)
        idxs = np.flatnonzero(mask)
        
        if len(idxs) > 1:
            start, end = idxs[0], idxs[-1]
            return (start + end) // 2 
        else:
            return idxs[0]

        best_idx = int(np.argmax(ranges))

        return best_idx

        # if ranges.size == 0:
        #     return (0 + ranges.size - 1) // 2

        # L = ranges.size
        # center_local = (L - 1) / 2.0
        # idxs = np.arange(L)
        
        # sigma = max(3.0, L / 6.0)
        # w = np.exp(-0.5 * ((idxs - center_local) / sigma) ** 2)
        # score = ranges * w
        # local_idx = int(np.argmax(score))
        # return local_idx


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges, data.angle_min, data.angle_max, data.angle_increment)

        # num_scan = len(proc_ranges)
        # self.get_logger().info(f"num_scan={num_scan}")

        # self.get_logger().info(f"closest_point={closest_idx},closest_dist = {closest_dist}")

        # self.get_logger().info(f"left={left},right = {right}")

        # i=0
        # for val in proc_ranges:
        #     self.get_logger().info(f"proc_ranges[{i}]={val}")
        #     i += 1

        #3. disparity_extender
        self.apply_disparity_extender(proc_ranges, data.angle_increment)

        # proc_ranges[left : right + 1] = 0.0

        # i=0
        # for val in proc_ranges:
        #     self.get_logger().info(f"proc_ranges[{i}]={val}")
        #     i += 1

        #4. Find max length gap 
        # gap_start, gap_end = self.find_max_gap(free_space)

        #5. Find the best point in the gap 
        best_idx = self.find_best_point(proc_ranges)

        # self.get_logger().info(f"best_idx = {best_idx}")

        steer = data.angle_min + (self.fov_min_id + best_idx) * data.angle_increment

        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        #Publish Drive message
        steer_deg = math.degrees(steer)

        # self.get_logger().info(f"Central range={data.ranges[540]:+.1f}")
        # self.get_logger().info(f"closest_dist={closest_dist:+.1f}, best_idx={best_idx}, best_range={data.ranges[best_idx + self.fov_min_id]}, steer={math.degrees(steer):+.1f}°\n")

        if steer_deg < self.small_turn_deg:
            speed = self.speed_hi
        elif steer_deg < self.mid_turn_deg:
            speed = self.speed_md
        else:
            speed = self.speed_lo

        # self.get_logger().info(f"speed={speed}\n")
        
        self.publish_drive(steer, speed)


    def publish_drive(self, steering_angle_rad, speed):

        msg = AckermannDriveStamped()

        msg.drive.steering_angle = steering_angle_rad
        msg.drive.speed = speed
        self.drive_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()