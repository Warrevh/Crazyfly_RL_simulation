from os import sync
import time
import gym
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from RLEnvironment import RLEnvironment, getAction

seed = 42

env = RLEnvironment(gui=True)
obs = env.reset


obs, info = env.reset(seed=4, options={})
start = time.time()

tot_reward = 0


for i in range(100000):

    action = np.array([[1,0]]) #getAction._getActionSquare(i) #

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
