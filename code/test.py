from os import sync
import time
import gym
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from RLEnvironment import RLEnvironment, getAction


parameters = {
    #env parameters
    'initial_xyzs': np.array([[0.5,0.5,0.2]]),
    'ctrl_freq': 60,
    'Target_pos': np.array([4,0.5,0.2]),
    'episode_length': 60,
    #Learning rate
    'Learning_rate': 0.0005,
    'Learning_rate_decay': -0.005,
    #Reward
    'Target_reward': -700,
    #evaluation callback
    'eval_freq': 10,

    'Total_timesteps': int(10e6),
}

seed = 42

env = RLEnvironment(parameters=parameters, gui=True)
obs = env.reset


obs, info = env.reset(seed=4, options={})
start = time.time()

tot_reward = 0


for i in range(10):

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
print(np.linalg.norm([2,3]))
