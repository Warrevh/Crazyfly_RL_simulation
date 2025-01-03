import rclpy
from rclpy.node import Node
from crazyflie_py import Crazyswarm
from std_msgs.msg import String

class CrazyflieNode(Node):
    def __init__(self,cf,timeHelper):
        super().__init__('crazyflie_node')

        self.cf = cf
        self.timeHelper = timeHelper
        # State management
        self.state = 'idle'  # Possible states: 'idle', 'takeoff', 'hover', 'land', 'goTo'

        self.z = 0.20

        self.cf.takeoff(targetHeight=self.z, duration=2.0)
        self.timeHelper.sleep(2.0)
        self.get_logger().info("Takeoff complete.")
        
        # Subscriber for commands
        self.command_subscriber = self.create_subscription(
            String,
            'drone_command',
            self.command_callback,
            10
        )

        # Timer to run the control loop
        self.timer = self.create_timer(1/60, self.control_loop)
        self.get_logger().info("Crazyflie node initialized and ready for commands.")

    def command_callback(self, msg):
        """Callback function to process received commands."""
        command = msg.data.lower()
        self.get_logger().info(f"Received command: {command}")

        if command == 'emergency':
            self.state = 'emergency'
        elif command == 'takeoff':
            self.state = 'takeoff'
        elif command.startswith('hover'):
            # Parse hover command, e.g., "hover 1.0 1.0"
            try:
                _, x, y= command.split()
                self.target_position = [float(x), float(y)]
                self.state = 'hover'
            except ValueError:
                self.get_logger().warn("Invalid hover command. Use 'hover x y'")

        elif command.startswith('goto'):
            # Parse hover command, e.g., "goto 1.0 1.0"
            try:
                _, x, y = command.split()
                self.target_position = [float(x), float(y)]
                self.state = 'goto'
            except ValueError:
                self.get_logger().warn("Invalid hover command. Use 'goto x y'")
        elif command == 'land':
            self.state = 'land'
        elif command == 'stop':
            self.state = 'idle'
        else:
            self.get_logger().warn(f"Unknown command: {command}")

    def control_loop(self):
        """Control loop that performs actions based on the current state."""
        if self.state == 'emergency':
            self.cf.emergency()
            self.get_logger().info("emergency")

        elif self.state == 'takeoff':
            self.cf.takeoff(targetHeight=self.z, duration=2.0)
            self.timeHelper.sleep(2.0)
            self.get_logger().info("Takeoff complete.")

        elif self.state == 'hover':
            x, y = self.target_position
            self.cf.goTo([x, y, self.z], yaw=0.0, duration=2.0)
            self.timeHelper.sleep(2.0)
            self.get_logger().info(f"Hovering at position: {self.target_position}")

        elif self.state == 'goto':
            x, y = self.target_position
            self.cf.goTo([x, y, self.z], yaw=0.0, duration=0.5)
            self.timeHelper.sleep(1/120)
            self.get_logger().info(f"Hovering at position: {self.target_position}")

        elif self.state == 'land':
            self.cf.land(targetHeight=0.0, duration=2.0)
            self.timeHelper.sleep(2.0)
            self.state = 'idle'
            self.get_logger().info("Landing complete. Returning to idle state.")

        elif self.state == 'idle':
            self.get_logger().info("Idle state. Waiting for commands...")


def main(args=None):


    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    node = CrazyflieNode(cf,timeHelper)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()