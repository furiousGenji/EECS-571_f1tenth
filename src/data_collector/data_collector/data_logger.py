import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import csv
from datetime import datetime

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.get_logger().info('🟢 Data Logger Node Started')

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, '/drive', self.drive_callback, 10)

        # Prepare CSV file
        self.filename = f"/home/russell/sim_ws/data_log/data_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['timestamp', 'angle_min', 'angle_max', 'speed', 'steering_angle', 'ranges'])

        self.latest_drive = None

    def scan_callback(self, msg):
        if self.latest_drive:
            self.writer.writerow([
                self.get_clock().now().to_msg().sec,
                msg.angle_min,
                msg.angle_max,
                self.latest_drive.drive.speed,
                self.latest_drive.drive.steering_angle,
                list(msg.ranges)
            ])

    def drive_callback(self, msg):
        self.latest_drive = msg

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f'💾 Data saved to {self.filename}')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🟥 Shutting down Data Logger...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
