#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
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

        self.odom_subscriber = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)

        self.start_line_x = -3.85         
          
        self.crossing_ready = True       
        self.prev_x = None
        self.prev_y = None
        self.lap_start_time = None
        self.lap_count = 0

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
        self.car_half_width = 0.22

        self.total_compute_time = 0
        self.compute_count = 0


    def odom_callback(self, msg):
        """
        Detect when the car crosses the start/finish line and compute lap time.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            return

        crossed = (
            self.prev_x < self.start_line_x <= x and
            y < 0.82 and y > -1
        )

        if crossed and self.crossing_ready:
            now = time.time()

            if self.lap_start_time is None:
                self.lap_start_time = now
                self.get_logger().info("Lap timer started!")
            else:
                lap_time = now - self.lap_start_time
                self.lap_count += 1

                self.get_logger().info(
                    f"Lap {self.lap_count} completed! Time = {lap_time:.3f} sec"
                )

                # Reset timer for next lap
                self.lap_start_time = now

            self.crossing_ready = False

        if abs(x - self.start_line_x) > 0.5:
            self.crossing_ready = True

        self.prev_x = x
        self.prev_y = y


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

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        # best_len = 0
        # best_start = 0
        # best_end = -1

        # i = 0
        # while i < len(free_space_ranges):
            
        #     while i < len(free_space_ranges) and free_space_ranges[i] <= 0:
        #         i += 1

        #     if i >= len(free_space_ranges):
        #         break

        #     start = i

        #     while i < len(free_space_ranges) and free_space_ranges[i] > 0:
        #         i += 1
            
        #     end = i - 1

        #     gap_len = end - start + 1
        #     if gap_len > best_len:
        #         best_len = gap_len
        #         best_start = start
        #         best_end = end 

        # return best_start, best_end

        valid = free_space_ranges > 0.0
        if not np.any(valid):
            return 0, max(0, len(free_space_ranges) - 1)

        edges = np.diff(valid.astype(np.int8))
        starts = np.where(edges == 1)[0] + 1
        ends   = np.where(edges == -1)[0]
        if valid[0]:
            starts = np.r_[0, starts]
        if valid[-1]:
            ends = np.r_[ends, len(valid) - 1]

        lengths = ends - starts + 1
        best = int(np.argmax(lengths))
        return int(starts[best]), int(ends[best])


    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        segment = ranges[start_i:end_i + 1]

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


    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """

        # t0 = time.perf_counter()

        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges, data.angle_min, data.angle_max, data.angle_increment)

        # num_scan = len(proc_ranges)
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

        # i=0
        # for val in free_space:
        #     self.get_logger().info(f"free_space[{i}]={val}")
        #     i += 1

        #4. Find max length gap 
        gap_start, gap_end = self.find_max_gap(free_space)

        #5. Find the best point in the gap 
        best_idx = self.find_best_point(gap_start, gap_end, free_space)

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

        # t1 = time.perf_counter()
        # dt = t1 - t0 

        # self.total_compute_time += dt
        # self.compute_count += 1

        # if self.compute_count % 5000 == 0:
        #     avg_ms = (self.total_compute_time / self.compute_count) * 1000.0
            # self.get_logger().info(f"Average compute time = {avg_ms:.3f} ms/step")


    def publish_drive(self, steering_angle_rad, speed):

        msg = AckermannDriveStamped()

        msg.drive.steering_angle = steering_angle_rad

        # self.get_logger().info(f"speed={speed}")

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