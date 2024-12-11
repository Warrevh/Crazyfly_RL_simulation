#!/usr/bin/env python

import sys
import time
import numpy as np
from pathlib import Path
from crazyflie_py import Crazyswarm

from stable_baselines3 import SAC

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf_transformations

from crazyflie_RL.droneposition import DronePosition
from crazyflie_RL.rl_model import RlModel

def main():
    drone_id = 'cf20'

    #model_path = "data/SAC_save-12.07.2024_22.03.53/best_model.zip"
    #model = RlModel(model_path)
    """
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    z = 0.2
    if len(sys.argv)>1:
        z = float(sys.argv[1])
    cf.takeoff(targetHeight=z, duration=(z+0.5))
    timeHelper.sleep((5))
    cf.land(targetHeight=0.04, duration=1.5)
    timeHelper.sleep((2))

    """

    rclpy.init()

    node = DronePosition(drone_id)

    node.destroy_node()
    rclpy.shutdown()
    
    """
    pos = np.zeros((1,3))
    vel = np.zeros((1,3))
    rpy = np.zeros((1,3))
    ang_v = np.zeros((1,3))

    obs = {
        "Position": np.array([pos[i,:] for i in range(1)]).astype('float32'),
        "Velocity": np.array([vel[i,:] for i in range(1)]).astype('float32'),
        "rpy": np.array([rpy[i,:] for i in range(1)]).astype('float32'),
        "ang_v": np.array([ang_v[i,:] for i in range(1)]).astype('float32'),
    }

    """

    print('test')



if __name__ == "__main__":
    main()

"""
class RlModel():
    def __init__(self,model_path):

        self.model = SAC.load(Path(__file__).parent / model_path)

    def get_action(self,obs):
        action, _states = self.model.predict(obs,deterministic=True)   

        return action    

   
class DronePosition(Node):
    def __init__(self,drone_id):
        super().__init__('drone_position_node')
        self.drone_id = drone_id

        # Create a buffer and listener for TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(1.0/240.0 , self.get_drone_position)
        self.prev_time = None

        self.prev_pos = None
        self.prev_rpy = None
        self.pos = np.zeros((1,3))
        self.vel = np.zeros((1,3))
        self.rpy = np.zeros((1,3))
        self.ang_v = np.zeros((1,3))
        
        self.obs = None

        model_path = "data/SAC_save-12.07.2024_22.03.53/best_model.zip"
        self.model = RlModel(model_path)

    def get_drone_position(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform('world', self.drone_id, rclpy.time.Time())
            
            position = transform.transform.translation
            quaternion = transform.transform.rotation
            current_time = self.get_clock().now()

            (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
            )

            self.pos = np.array([[position.x,position.y,position.z]])
            self.rpy = np.array([[roll,pitch,yaw]])

            if self.prev_pos is not None and self.prev_time is not None:
                # Calculate time difference (seconds)
                time_diff = (current_time - self.prev_time).nanoseconds / 1e9

                self.vel = (self.pos - self.prev_pos)/time_diff

                if self.prev_rpy is not None:
                    self.ang_v = (self.rpy - self.prev_rpy)/time_diff

            self.prev_pos = self.pos
            self.prev_rpy = self.rpy
            self.prev_time = current_time

            self.obs = {
                "Position": np.array([self.pos[i,:] for i in range(1)]).astype('float32'),
                "Velocity": np.array([self.vel[i,:] for i in range(1)]).astype('float32'),
                "rpy": np.array([self.rpy[i,:] for i in range(1)]).astype('float32'),
                "ang_v": np.array([self.ang_v[i,:] for i in range(1)]).astype('float32'),
            }

            #print(self.vel)

            self.get_action_node()

            
            #self.get_logger().info(f"Position: x={position.x}, y={position.y}, z={position.z}")
            #self.get_logger().info(f"Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to get transform: {str(e)}")

        return self.obs
    
    def get_action_node(self):

        print(self.model.get_action(self.obs))

"""