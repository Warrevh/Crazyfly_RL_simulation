import time
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

from RLEnvironment import RLEnvironment, getAction


#this code is used for testing the evironment

def log_obs(obs):
        flattened_obs = [
            obs["Position"][0][0], obs["Position"][0][1], obs["Position"][0][2],  # Position (x, y, z)
            obs["Velocity"][0][0], obs["Velocity"][0][1], obs["Velocity"][0][2],  # Velocity (x, y, z)
            obs["rpy"][0][0], obs["rpy"][0][1], obs["rpy"][0][2],                # Roll, Pitch, Yaw
            obs["ang_v"][0][0], obs["ang_v"][0][1], obs["ang_v"][0][2]          # Angular velocity (x, y, z)
        ]

        return flattened_obs


parameters = {
    #env parameters
    'initial_xyzs': np.array([[4.5,3.5,0.2]]),
    'random_initial_pos': True,
    'obs_noise': True,
    'ctrl_freq': 240,
    'Target_pos': np.array([2.5,2,0.2]),
    'episode_length': 60,
    #Learning
    'Learning_rate': 0.001,
    'learning_starts': 100000,
    'batch_size':256,
    'use_sde':False ,
    'sde_sample_freq': -1,
    #Reward
    'Target_reward': 1500,
    #Reward Function
    'Rew_distrav_fact': 0,
    'Rew_disway_fact': 0.01,
    'Rew_step_fact': 0,
    'Rew_direct_fact': 100,
    'Rew_angvel_fact': 10,
    'Rew_collision': -100,
    'Rew_terminated': 1000,
    #evaluation callback
    'eval_freq': 1, #"epsisodes" (eval_freq*(epsiode_length*ctrl_freq))
    'eval_episodes': 5,
    #observation !!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'position': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'velocity': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'rpy': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'ang_v': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'prev_act':False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    #train
    'number_of_env': 1,
    'Total_timesteps': int(3e6),
    'train_freq': 1,
    'gradient_steps': -1,
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))+self.Rew_disway_fact*max(0,2-np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis)-self.Rew_angvel_fact*(np.sum((self.angvel-prev_angvel)**2)))',
    'parent_model': "none"
}

seed = 42

env = RLEnvironment(parameters=parameters, gui=True)
obs = env.reset

obs, info = env.reset(seed=4, options={})
start = time.time()

tot_reward = 0
all_obs = []
rewards = []

for i in range(100000):

    action = np.array([[-1,0]]) #getAction._getRandomAction()#np.array([[1,0]]) #getAction._getActionSquare(i)
    """
    if i < 5000:
         action = np.array([[1,0]])
    else:
         action = np.array([[-1,0]])
    """
         
    obs, reward, terminated, truncated, info = env.step(action)
    print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

    all_obs.append(log_obs(obs))
    tot_reward += reward
    print(reward)
    rewards.append(reward)

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

x_positions = [obs[3] for obs in all_obs]
steps = list(range(1, len(x_positions) + 1))  # x-axis: steps (1 to the length of speed)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(steps, x_positions, marker='o', linestyle='-', color='blue', label='Speed over Steps')

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Speed')
plt.title('Speed vs. Steps')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

