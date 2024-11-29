import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class RLEnvironment(BaseRLAviary):

    def __init__(self, 
                 drone_model = DroneModel.CF2X,
                 num_drones = 1,
                 neighbourhood_radius = np.inf,
                 initial_rpys=None,
                 physics = Physics.PYB,
                 pyb_freq = 240,
                 gui=False,
                 record=False,
                 obs = ObservationType.KIN,
                 act = ActionType.PID,
                 parameters = None
                 ):
        
        self.INITIAL_XYZS = parameters['initial_xyzs'] #np.array([[4.5,3.5,0.2]]) #np.array([[-1.5,-1.5,0.2]])
        self.CTRL_FREQ = parameters['ctrl_freq']
        self.Rew_distrav_factor = parameters['Rew_distrav_factor']
        self.Rew_disway_factor = parameters['Rew_disway_factor']

        self.act2d = True

        self.waypoint = False
        self.smallWaypoints_POS = np.array([[1,0.5,0.2],[2,0.5,0.2]])
        self.smallWaypoint_RAD = 0.1 

        self.TARGET_POS = parameters['Target_pos'] #np.array([2.5,2,0.2]) #np.array([0.15,2.5,0.2])
        self.TARGET_RAD = 0.1
        
        super().__init__(drone_model, 
                         num_drones, 
                         neighbourhood_radius, 
                         self.INITIAL_XYZS, 
                         initial_rpys, 
                         physics, 
                         pyb_freq, 
                         self.CTRL_FREQ, 
                         gui, 
                         record, 
                         obs, 
                         act
                         )
        
        
        self.EPISODE_LEN_SEC = parameters['episode_length']
        
        self.reward_state = self._getDroneStateVector(0)[0:2]

    def step(self,action):

        obs, reward, terminated, truncated, info = super().step(action)
        
        if self._getCollision(self.DRONE_IDS[0]):
            reward = self._computeReward()

            self.reset_drone()

            action = np.array([[0,0]])

            obs, _, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def reset(self,seed : int = None,options : dict = None):

        self.ctrl[0].reset()

        return super().reset(seed,options)

    def _computeReward(self):
       
        prev_state = self.reward_state[0:2]
        self.reward_state = self._getDroneStateVector(0)[0:2]

        ret = -self.Rew_distrav_factor*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))-self.Rew_disway_factor*(np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4) #-1 each step
        #-0.01*(abs(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))) #negative reward for distance travelled
         #negative reward for distance to target

        if self._getCollision(self.DRONE_IDS[0]):
            ret = -100 #reward for hitting wall
        elif self._computeTerminated():
            ret = 1000 #reward for reaching target
        elif self.waypoint and self.hit_waypoint():
            ret = 100 #reward for reaching waypoint

        return ret
    
    def _computeTerminated(self):

        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS[0:2]-state[0:2]) < self.TARGET_RAD:
            print("Terminated")
            return True
        else:
            return False

    def _computeTruncated(self):

        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 5 or abs(state[1]) > 5 or state[2] > 5 # Truncate when the drone is too far away
             or abs(state[7]) > .9 or abs(state[8]) > .9 # Truncate when the drone is too tilted
            ):
            print("tillted/outofbound")
            return True
        
    
        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:

            print("Timeout")
            return True

    
        
            """
        if self._getCollision(self.DRONE_IDS[0]):

            print("collision")
            return True
            """
        
        else:
            return False
        
    def _addObstacles(self):


        self.ROOMID = p.loadURDF("recourses/Room_big/urdf/Room_big.urdf",
                   (0,4,0),
                   physicsClientId=self.CLIENT,
                   useFixedBase = True
                )   
        
        self.LOGOID = p.loadURDF("recourses/Logo/logo_urdf/urdf/logo_urdf.urdf",
                   (0.8,1,-0.001),
                   physicsClientId=self.CLIENT,
                   useFixedBase = True
                )
        if self.waypoint:
            self.smallWaypoints_IDs = np.array([])
            for i in range(self.smallWaypoints_POS.shape[0]):
                id = createWaypoint._makeSmallWaypoint(self.smallWaypoints_POS[i,:],self.smallWaypoint_RAD,self.CLIENT)
                self.smallWaypoints_IDs = np.append(self.smallWaypoints_IDs, id)

            self.hit_smallWaypoints_POS = self.smallWaypoints_POS
            self.hit_smallWaypoints_IDs = self.smallWaypoints_IDs

        self.TARGET_ID = createWaypoint._makeTarget(self.TARGET_POS,self.TARGET_RAD,self.CLIENT)
        
    def _getCollision(self,obj1):

        contact_points = p.getContactPoints(obj1, physicsClientId=self.CLIENT)

        if len(contact_points) > 0:
            return True
        else:
            return False

    def _preprocessAction(self,action):

        if self.act2d == True and self.ACT_TYPE == ActionType.PID:
            self.action_buffer.append(action)
            rpm = np.zeros((self.NUM_DRONES,4))
            for k in range(action.shape[0]):
                target = action[k, :]
                state = self._getDroneStateVector(k)
                res, pos_err, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                            cur_pos=state[0:3],
                                                            cur_quat=state[3:7],
                                                            cur_vel=state[10:13],
                                                            cur_ang_vel=state[13:16],
                                                            target_pos=state[0:3]+0.1*np.array([target[0],target[1],0])
                                                            )
                rpm[k,:] = res
            #print(state[0:3]+0.01*np.array([target[0],target[1],0]))
            return rpm
        
        else:
            return super()._preprocessAction(action)

    def _actionSpace(self):
        
        if self.ACT_TYPE ==ActionType.PID and self.act2d == True:
            size = 2
        else: 
            return super()._actionSpace(self)
        
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):

        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE ==ActionType.PID and self.act2d == True:
            """
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[-5,-5,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)]) #np.array([[-5,-5,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[5,5,5,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)]) #np.array([[5,5,5,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            
            ############################################################
            """
            lo = -np.inf
            hi = np.inf

            pos_lo = np.array([[-5,-5,0] for i in range(self.NUM_DRONES)])
            pos_hi = np.array([[5,5,3] for i in range(self.NUM_DRONES)])
            vel_lo = np.array([[-50,-50,-50] for i in range(self.NUM_DRONES)])
            vel_hi = np.array([[50,50,50] for i in range(self.NUM_DRONES)])
            rpy_lo = np.array([[-np.pi,-np.pi,-np.pi] for i in range(self.NUM_DRONES)])
            rpy_hi = np.array([[np.pi,np.pi,np.pi] for i in range(self.NUM_DRONES)])
            ang_v_lo = np.array([[-2,-2,-2] for i in range(self.NUM_DRONES)])
            ang_v_hi = np.array([[2,2,2] for i in range(self.NUM_DRONES)])

            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                act_lower_bound = np.array([[act_lo,act_lo] for i in range(self.NUM_DRONES)])
                act_upper_bound = np.array([[act_hi,act_hi] for i in range(self.NUM_DRONES)])



            ret = spaces.Dict({
                "Position": spaces.Box(low=pos_lo, high=pos_hi,dtype=np.float32),
                #"Velocity": spaces.Box(low=vel_lo, high=vel_hi,dtype=np.float32),
                #"rpy": spaces.Box(low=rpy_lo, high=rpy_hi,dtype=np.float32),
                #"ang_v": spaces.Box(low=ang_v_lo, high=ang_v_hi,dtype=np.float32),
                "prev_act": spaces.Box(low=act_lower_bound, high=act_upper_bound,dtype=np.float32)
            })

            return ret

            


        else:
            return super()._observationSpace(self)
        
    def _computeInfo(self):

        return {"answer": 42} 
    
    def _computeObs(self):

        if self.OBS_TYPE == ObservationType.KIN and self.ACT_TYPE ==ActionType.PID and self.act2d == True:

            """
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,) #[obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            """
            pos = np.zeros((self.NUM_DRONES,3))
            vel = np.zeros((self.NUM_DRONES,3))
            rpy = np.zeros((self.NUM_DRONES,3))
            ang_v = np.zeros((self.NUM_DRONES,3))
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                pos[i,:]=obs[0:3]
                vel[i,:]=obs[10:13]
                rpy[i,:]=obs[7:10]
                ang_v[i,:]=obs[13:16]

            for i in range(self.ACTION_BUFFER_SIZE):
                act = np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])
            ret = {
                "Position": np.array([pos[i,:] for i in range(self.NUM_DRONES)]).astype('float32'),
                #"Velocity": np.array([vel[i,:] for i in range(self.NUM_DRONES)]).astype('float32'),
                #"rpy": np.array([rpy[i,:] for i in range(self.NUM_DRONES)]).astype('float32'),
                #"ang_v": np.array([ang_v[i,:] for i in range(self.NUM_DRONES)]).astype('float32'),
                "prev_act": act.astype('float32')
            }

            return ret
        else:
            return super()._computeObs(self)
    
    def reset_drone(self):

        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.quat[0,3] = 1
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))

        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], self.INIT_XYZS[0,:],p.getQuaternionFromEuler(self.INIT_RPYS[0,:]), physicsClientId=self.CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0],self.vel[0,:],self.ang_v[0,:], physicsClientId=self.CLIENT)

        self.ctrl[0].reset()

    def hit_waypoint(self):
        
        for i in range(self.hit_smallWaypoints_POS.shape[0]):
            if np.linalg.norm(self.hit_smallWaypoints_POS[i][0:2]-self.reward_state[0:2]) < self.smallWaypoint_RAD:
                self.hit_smallWaypoints_POS = np.delete(self.hit_smallWaypoints_POS,i, axis=0)
                p.removeBody(int(self.hit_smallWaypoints_IDs[i]),physicsClientId=self.CLIENT)
                self.hit_smallWaypoints_IDs = np.delete(self.hit_smallWaypoints_IDs,i, axis=0)

                return True
        
        return False

class getAction():

    def _getRandomAction():
        action = np.random.uniform(-1, 1, size=(1, 2))

        return action
    
    def _getActionSquare(i):
        speed = 1
        i = i%1000
        
        if i in range(0,251):
            action = np.array([[speed,0]])
        
        elif i in range(251,501):
            action = np.array([[0,speed]])

        elif i in range(501,751):
            action = np.array([[-speed,0]])

        elif i > 750:
            action = np.array([[0,-speed]])

        return action
    
class createWaypoint():

    def _makeSmallWaypoint(pos,radius,CLIENT):

        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius=radius,
                                    rgbaColor=[1, 0, 0, 0.3],
                                    visualFramePosition=[0,0,0],
                                    physicsClientId=CLIENT)
        
        body_id = p.createMultiBody(
                      baseVisualShapeIndex=visual_id,
                      basePosition = pos,
                      physicsClientId=CLIENT)
        return body_id

    def _makeTarget(pos,radius,CLIENT):

        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                    radius=radius,
                                    rgbaColor=[0, 1, 0, 0.3],
                                    visualFramePosition=[0,0,0],
                                    physicsClientId=CLIENT)
        
        body_id = p.createMultiBody(
                      baseVisualShapeIndex=visual_id,
                      basePosition = pos,
                      physicsClientId=CLIENT)
        return body_id
