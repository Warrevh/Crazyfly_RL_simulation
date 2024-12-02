from os import sync
import time
import gym
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from RLEnvironment import RLEnvironment, getAction


parameters = {
    #env parameters
    'initial_xyzs': np.array([[1,0.5,0.2]]),
    'ctrl_freq': 240,
    'Target_pos': np.array([0.5,0.5,0.2]),
    'episode_length': 40,
    #Learning rate
    'Learning_rate': 0.0005,
    'Learning_rate_decay': -0.005,
    #Reward
    'Target_reward': -0,
    #Reward Function
    'Rew_distrav_fact': 0,
    'Rew_disway_fact': 0,
    'Rew_step_fact': 0,
    'Rew_tardis_fact': 1000,
    #evaluation callback
    'eval_freq': 5, #"epsisodes" (eval_freq*(epsiode_length*ctrl_freq))
    #observation !!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'position': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'velocity': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'rpy': False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'ang_v': False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'prev_act':False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    #train
    'Total_timesteps': int(30e6),
    'train_freq': int(240//2),
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))-self.Rew_disway_fact*(np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis))'
}

seed = 42

env = RLEnvironment(parameters=parameters, gui=True)
obs = env.reset


obs, info = env.reset(seed=4, options={})
start = time.time()

tot_reward = 0


for i in range(10000):

    action = np.array([[1,0]]) #getAction._getRandomAction()#np.array([[1,0]]) #getAction._getActionSquare(i)

    obs, reward, terminated, truncated, info = env.step(action)
    print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

    tot_reward += reward

    if env._getCollision(env.DRONE_IDS[0]):
        #env.reset()
        print("COLISSION")

    env.render()
    if terminated:
        print(tot_reward)
        tot_reward = 0
        obs = env.reset(seed=42, options={})

    time.sleep(0.0)

env.close()

print(tot_reward)

action = env._actionSpace()
state = env._observationSpace()
print(action)
print(state)