#!/usr/bin/env python3
import os
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32
import torch
import torch.nn as nn
import numpy as np

import time



#  MLP MODEL DEFINITITION
class SingleHeadMLP(nn.Module):
    def __init__(self, in_dim=182, hidden=256, out_dim=3):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)

#  ROS2 NODE

class MLPControllerNode(Node):
    def __init__(self):
        super().__init__('mlp_controller_node')

        self.total_compute_time = 0
        self.compute_count = 0

        # ----- Parameters -----
        # Path to .pth model file (you can override via ROS2 param)
        self.declare_parameter('model_path', 'resource/mlp_safe_final.pt')
        self.declare_parameter('input_dim', 182)
        self.declare_parameter('max_steering_angle', 0.34)
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('min_speed', 0.5)
        self.declare_parameter('confidence_speed_scale', 0.7)  # scale speed by conf^scale
        self.declare_parameter('scan_length', 180)
        self.declare_parameter('ftg_topic', '/drive')          # FTG output topic
        self.declare_parameter('scan_topic', '/scan')          # LiDAR
        self.declare_parameter('output_topic', '/drive_mlp')   # corrected drive cmd

        self.lidar_points = self.get_parameter('scan_length').get_parameter_value().integer_value
        # model_path = self.get_parameter('model_path').get_parameter_value().string_value
        pkg_share = get_package_share_directory('mlp_controller')
        model_path = os.path.join(pkg_share, self.get_parameter('model_path').value)
        
        input_dim = self.get_parameter('input_dim').get_parameter_value().integer_value

	# -------- Load model + normalization --------
        self.device = torch.device("cpu")

	# Build model
        input_dim = self.lidar_points + 2
        self.model = SingleHeadMLP(in_dim=input_dim).to(self.device)
        self.model.eval()

	# Load weights
        if not os.path.isfile(model_path):
            self.get_logger().warn(
                f"Model file not found at '{model_path}'. "
                "Node will run, but output will be zero corrections."
            )
            self.model_loaded = False
        else:
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
                self.mean = np.array(ckpt["mean"], dtype=np.float32)
                self.std  = np.array(ckpt["std"], dtype=np.float32)


                self.model_loaded = True
                self.get_logger().info(f"Loaded MLP model from: {model_path}")

            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
                self.model_loaded = False

	# Last FTG drive message
        self.last_drive_msg: Optional[AckermannDriveStamped] = None

        # ----- Subscribers & Publisher -----
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            10
        )

        self.drive_sub = self.create_subscription(
            AckermannDriveStamped,
            self.get_parameter('ftg_topic').get_parameter_value().string_value,
            self.drive_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.get_parameter('output_topic').get_parameter_value().string_value,
            10
        )
        self.conf_pub = self.create_publisher(Float32, '/mlp_confidence', 10)

        # self.get_logger().info("MLPControllerNode initialized.")


    # Callbacks

    def drive_callback(self, msg: AckermannDriveStamped):

        self.last_drive_msg = msg

    def scan_callback(self, scan_msg: LaserScan):

        t0 = time.perf_counter()

        if self.last_drive_msg is None:
            return

        # Preprocess LiDAR
        scan_length = self.get_parameter('scan_length').get_parameter_value().integer_value
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        max_range = scan_msg.range_max if scan_msg.range_max > 0.0 else 10.0
        ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=0.0)

        if len(ranges) > scan_length:
            ranges = ranges[:scan_length]
        elif len(ranges) < scan_length:
            pad_len = scan_length - len(ranges)
            ranges = np.pad(ranges, (0, pad_len), mode='constant', constant_values=max_range)

        # Normalize LiDAR roughly [0,1]
        ranges_norm = ranges / max_range

        # Get FTG steering & speed
        ftg_steer = self.last_drive_msg.drive.steering_angle
        ftg_speed = self.last_drive_msg.drive.speed

        # Build input vector [1080 lidar + steering + speed]
        input_vec = np.concatenate([
            ranges_norm,
            np.array([ftg_steer], dtype=np.float32),
            np.array([ftg_speed], dtype=np.float32)
        ], axis=0)

        if input_vec.shape[0] != self.get_parameter('input_dim').get_parameter_value().integer_value:
            # self.get_logger().warn(
            #     f"Input dim mismatch: got {input_vec.shape[0]}, "
            #     f"expected {self.get_parameter('input_dim').get_parameter_value().integer_value}"
            # )
            return

        # Run through model
        steering_corr, speed_corr, confidence = self.run_model(input_vec)

        # Publish confidence for arbiter
        self.conf_pub.publish(Float32(data=confidence))

        # Build corrected command
        corrected_msg = self.build_corrected_drive_msg(
            scan_msg=scan_msg,
            base_drive=self.last_drive_msg,
            steering_corr=steering_corr,
            speed_corr=speed_corr,
            confidence=confidence
        )

        self.drive_pub.publish(corrected_msg)

        t1 = time.perf_counter()
        dt = t1 - t0 

        self.total_compute_time += dt
        self.compute_count += 1

        if self.compute_count % 5000 == 0:
            avg_ms = (self.total_compute_time / self.compute_count) * 1000.0
            self.get_logger().info(f"Average compute time = {avg_ms:.3f} ms/step")

    # Model inference    
    def run_model(self, input_vec: np.ndarray):
    	if not self.model_loaded:
        	return 0.0, 0.0, 1.0

    	with torch.no_grad():
        	x = torch.from_numpy(input_vec).float().to(self.device)
        	x = x.unsqueeze(0)

        	out = self.model(x).squeeze()

        	steer_corr = out[0].item()
        	speed_corr = out[1].item()
        	conf       = out[2].item()

        	# self.get_logger().info(f"MLP Corrections -> steer: {steer_corr:.3f}, speed: {speed_corr:.3f}, conf: {conf:.3f}")

        	return steer_corr, speed_corr, conf

    # Build corrected command
    def build_corrected_drive_msg(
        self,
        scan_msg: LaserScan,
        base_drive: AckermannDriveStamped,
        steering_corr: float,
        speed_corr: float,
        confidence: float
    ) -> AckermannDriveStamped:

        max_steer = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        conf_speed_scale = self.get_parameter('confidence_speed_scale').get_parameter_value().double_value

        # Base values from FTG
        base_steer = base_drive.drive.steering_angle
        base_speed = base_drive.drive.speed

        # Apply corrections
        new_steer = base_steer + steering_corr
        new_speed = base_speed + speed_corr

        # Clip steering
        new_steer = float(max(-max_steer, min(max_steer, new_steer)))

        # Use confidence to modulate speed (low confidence => slow down)
        # Example: effective_speed = new_speed * (confidence ** conf_speed_scale)
        confidence_clamped = float(max(0.0, min(1.0, confidence)))
        effective_speed = new_speed * (confidence_clamped ** conf_speed_scale)

        # Clip speed
        effective_speed = float(max(min_speed, min(max_speed, effective_speed)))

        # Build new message
        out = AckermannDriveStamped()
        out.header = scan_msg.header  # sync with LiDAR timestamp
        out.drive.steering_angle = new_steer
        out.drive.speed = effective_speed

        return out


def main(args=None):
    rclpy.init(args=args)
    node = MLPControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
