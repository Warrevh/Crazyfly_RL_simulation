from stable_baselines3 import TD3
import numpy as np
import pickle
from datetime import datetime
import os

from RLEnvironment import RLEnvironment

#for running the model multiple episodes and collecting data
class Multiple_runs():
    def __init__(self,n_runs, env_parameters, store_path, model_type):
        self.n_runs = n_runs

        self.store_path = store_path
        current_time = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        self.file_path = str(self.store_path + f"/data_{self.n_runs}_runs_with_{model_type}_model_{current_time}.pkl")

        model_type_ = model_type + '_model.zip'
        model = os.path.join(store_path, model_type_)

        self.env_parameters = env_parameters
        self.env = RLEnvironment( parameters=env_parameters)
        self.model = TD3.load(model)
        self.all_runs_data = []

    def one_run(self):
        run_data = np.array([])
        obs, info = self.env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs,deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            run_data = np.append(run_data,self.box_data(obs, reward, action, terminated, truncated))
            if terminated or truncated:
                done = True

        return run_data
    
    def box_data(self,obs,reward,action,terminated, truncated):
        flattened_obs = [
            obs["Position"][0][0], obs["Position"][0][1], obs["Position"][0][2],  # Position (x, y, z)
            obs["Velocity"][0][0], obs["Velocity"][0][1], obs["Velocity"][0][2],  # Velocity (x, y, z)
            obs["rpy"][0][0], obs["rpy"][0][1], obs["rpy"][0][2],                # Roll, Pitch, Yaw
            obs["ang_v"][0][0], obs["ang_v"][0][1], obs["ang_v"][0][2]          # Angular velocity (x, y, z)
        ]

        step_data = {
            "obs": flattened_obs,
            "reward": reward,
            "action": action[0],
            "terminated": terminated,
            "truncated": truncated
        }
        return step_data
    
    def all_runs(self):
        for i in range(self.n_runs):
            self.all_runs_data.append(self.one_run())
            print("epsisode:", i)

        self.save_data()
        return(self.all_runs_data)

    def save_data(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.all_runs_data, f)

        print("Data saved")

parameters = {
    #env parameters
    'initial_xyzs': np.array([[4.5,3.5,0.2]]), #start position of the drone
    'random_initial_pos': True, #If this is True, the start position of the drone is randomized within an area at the start of each episode.
    'obs_noise': True, #add noise to the observation
    'ctrl_freq': 240, #frequency of the controller 
    'Target_pos': np.array([2.5,2,0.2]), #positon of the goal
    'episode_length': 60, #amount of seconds before the epsisode times out
    #Learning (for explenation see documentation of Stable Baselines3)
    'Learning_rate': 0.001,
    'learning_starts': 100000,
    'batch_size':256,
    'use_sde':False ,
    'sde_sample_freq': -1,
    #Reward
    'Target_reward': 15000, #traing stops once target reward is reached
    #Reward Function
    'Rew_distrav_fact': 0, #factor for the negative reward of travelled distance
    'Rew_disway_fact': 0.01, #factor for positive reward for distacne from the target (closser is higher reward, only at a distance of 2m)
    'Rew_step_fact': 0, #factor for negative reward of each step
    'Rew_direct_fact': 100, #factor for direction of travel (posetive of in direction of target)
    'Rew_angvel_fact': 5, #factor for changes in angular velocity (negative)
    'Rew_collision': -100, #reward for colliding into wall
    'Rew_terminated': 1000, #reward if the goal is reached
    #evaluation callback
    'eval_freq': 1, #"episodes" (eval_freq*(epsiode_length*ctrl_freq)) how often the agent is evaluated (in episodes)
    'eval_episodes': 5, #amount of evaluation episodes
    #observation !!!!!!! ADJUST MANUALY IN CODE !!!!!!! (what data to include in the observation space, True is inculded)
    'position': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'velocity': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'rpy': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'ang_v': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'prev_act':False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    #train
    'number_of_env': 1, #number of parallel environments (not propperly implemented)
    'Total_timesteps': int(3e6), #amount of steps to train for
    'train_freq': 1, #Update the model every ``train_freq`` steps
    'gradient_steps': -1, #amount of gradient steps, -1 is same amount as train_freq
    #metadata
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))+self.Rew_disway_fact*max(0,2-np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis)-self.Rew_angvel_fact*(np.sum((self.angvel-prev_angvel)**2)))',
    'parent_model': "none"
}
path = 'results/SAC_save-01.15.2025_00.49.25'
model_type = 'best' #final or best
n_runs = 100 #amount of times to test the model
runs = Multiple_runs(n_runs, parameters, path, model_type)
test = runs.all_runs()
