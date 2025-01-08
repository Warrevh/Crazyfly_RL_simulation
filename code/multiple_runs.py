from stable_baselines3 import TD3
import numpy as np
import pickle
from datetime import datetime
import os

from RLEnvironment import RLEnvironment

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
path = 'results/TD3_save-01.07.2025_16.13.26'
model_type = 'best'
n_runs = 50
runs = Multiple_runs(n_runs, parameters, path, model_type)
runs.all_runs()

"""
print(test)
print(len(test))
print(sum(d["reward"] for d in data))

total_reward = 0

for ep in test:
    reward = sum(d["reward"] for d in ep)
    print(reward)
    total_reward += reward


avg_rew = total_reward/len(test)
print(avg_rew)
"""