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
        # self.disp_thresh = 0.5
        # self.extend_bubble_radius = 30


    def preprocess_lidar(self, ranges, angle_min, angle_max, angle_increment):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # proc_ranges = ranges
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


    def disparity_extend(self, free_space_ranges, disp_thresh, angle_increment, car_half_width=0.22):

        length = len(free_space_ranges)
        for i in range(length - 1):
            d1, d2 = free_space_ranges[i], free_space_ranges[i + 1]
            if d1 <= 0.0 or d2 <= 0.0:
                continue
            if abs(d1 - d2) > disp_thresh:
                
                # dnear = min(d1, d2)
                # theta = math.atan2(car_half_width, max(dnear, 0.05))
                # k = int(max(1, math.ceil(theta / angle_increment)))
                k = 10
                if d1 < d2:
                    free_space_ranges[max(0, i - k + 1): i + 1] = 0.0
                else:
                    free_space_ranges[i + 1: min(length, i + 1 + k)] = 0.0
        return free_space_ranges
    

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        best_len = 0
        best_start = 0
        best_end = -1

        i = 0
        while i < len(free_space_ranges):
            
            while i < len(free_space_ranges) and free_space_ranges[i] <= 0:
                i += 1

            if i >= len(free_space_ranges):
                break

            start = i

            while i < len(free_space_ranges) and free_space_ranges[i] > 0:
                i += 1
            
            end = i - 1

            gap_len = end - start + 1
            if gap_len > best_len:
                best_len = gap_len
                best_start = start
                best_end = end 

        return best_start, best_end


    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        segment = ranges[start_i:end_i + 1]

        # local_idx = int(np.argmax(segment))
        # return start_i + local_idx

        if segment.size == 0:
            return (start_i + end_i) // 2

        L = segment.size
        center_local = (L - 1) / 2.0
        idxs = np.arange(L)
        
        sigma = max(3.0, L / 6.0)
        w = np.exp(-0.5 * ((idxs - center_local) / sigma) ** 2)
        score = segment * w
        local_idx = int(np.argmax(score))
        return start_i + local_idx

    # def extend_bubble(self, arr, start_i, end_i):

    #     left = max(0, start_i - self.extend_bubble_radius)
    #     right = min(len(arr) - 1, end_i + self.extend_bubble_radius)
    #     arr[left:right + 1] = 0.0
    #     return arr


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges, data.angle_min, data.angle_max, data.angle_increment)

        num_scan = len(proc_ranges)
        # self.get_logger().info(f"num_scan={num_scan}")
        
        #1. Find closest point to LiDAR
        closest_idx = int(np.argmin(proc_ranges))
        closest_dist = float(proc_ranges[closest_idx])

        # self.get_logger().info(f"closest_point={closest_idx},closest_dist = {closest_dist}")

        #2. Eliminate all points inside 'bubble' (set them to zero) 
        bubble = int(self.bubble_points + max(0, (1.0 / max(closest_dist, 0.3)) * 6)) # Extend the bubble range as car is getting closer to the obstacle
        left = max(0, closest_idx - bubble)
        right = min(proc_ranges.size - 1, closest_idx + bubble)

        # self.get_logger().info(f"left={left},right = {right}")

        free_space = proc_ranges.copy()
        free_space[left:right + 1] = 0.0 

        #Extend the bubble range
        # free_space = self.extend_bubble(free_space, left, right)
        

        # i=0
        # for val in free_space:
        #     self.get_logger().info(f"free_space[{i}]={val}")
        #     i += 1

        # self.disparity_extend(free_space, self.disp_thresh, data.angle_increment, 0.22)

        #3. Find max length gap 
        gap_start, gap_end = self.find_max_gap(free_space)

        #4. Find the best point in the gap 
        best_idx = self.find_best_point(gap_start, gap_end, free_space)

        steer = data.angle_min + (self.fov_min_id + best_idx) * data.angle_increment

        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        #Publish Drive message
        steer_deg = math.degrees(steer)

        # self.get_logger().info(f"Central range={data.ranges[540]:+.1f}")
        # self.get_logger().info(f"closest_dist={closest_dist:+.1f}, best_idx={best_idx}, best_range={data.ranges[best_idx + self.fov_min_id]}, steer={math.degrees(steer):+.1f}°\n")

        # if(closest_dist < 0.6):
        #     speed = 0.5
        # else:
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