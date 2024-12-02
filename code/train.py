from datetime import datetime
import os
import time
import gym
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.buffers import DictReplayBuffer

from RLEnvironment import RLEnvironment
from data_handling import Txt_File

parameters = {
    #env parameters
    'initial_xyzs': np.array([[4.5,3.5,0.2]]),
    'ctrl_freq': 240,
    'Target_pos': np.array([2.5,2,0.2]),
    'episode_length': 30,
    #Learning rate
    'Learning_rate': 0.0005,
    'Learning_rate_decay': -0.005,
    #Reward
    'Target_reward': 100000,
    #Reward Function
    'Rew_distrav_fact': 0,
    'Rew_disway_fact': 0.1,
    'Rew_step_fact': 0,
    'Rew_tardis_fact': 1,
    #evaluation callback
    'eval_freq': 1, #"epsisodes" (eval_freq*(epsiode_length*ctrl_freq))
    #observation !!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'position': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'velocity': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'rpy': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'ang_v': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'prev_act':False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    #train
    'Total_timesteps': int(1e6),
    'train_freq': 1,
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))-self.Rew_disway_fact*(np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis))'
}

eval_freq = parameters['eval_freq']*parameters['ctrl_freq']*parameters['episode_length']

train_giu = True

output_folder= 'results'

filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
if not os.path.exists(filename):
    os.makedirs(filename+'/')

Myparameters = Txt_File(filename)
Myparameters.save_parameters(parameters)

train_env = make_vec_env(lambda: RLEnvironment(parameters=parameters),n_envs=1, seed=0)

eval_env = RLEnvironment(parameters=parameters,gui = train_giu)

print('[INFO] Action space:', train_env.action_space)
print('[INFO] Observation space:', train_env.observation_space)

n_actions = train_env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=1 * np.ones(n_actions),theta=0.10, dt=1)

def lineair_decay(progress_remaining):
    progress_remaining
    return parameters['Learning_rate'] * np.exp(parameters['Learning_rate_decay'] * progress_remaining)


"""
model = DDPG.load("results/trained big box 2.0 save-11.21.2024_23.05.24/final_model.zip",train_env)
"""
model = DDPG('MultiInputPolicy',train_env,
             learning_rate=parameters['Learning_rate'],
             learning_starts=1,
             action_noise=action_noise,
             train_freq= (int(1), "step"), #int(eval_env.CTRL_FREQ//2)
             replay_buffer_class= DictReplayBuffer,
             verbose=1)

target_reward = parameters['Target_reward']

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                    verbose=1)
eval_callback = EvalCallback(eval_env,
                                callback_on_new_best=callback_on_best,
                                verbose=1,
                                n_eval_episodes= 2,
                                best_model_save_path=filename+'/',
                                log_path=filename+'/',
                                eval_freq=eval_freq,
                                deterministic=True,
                                render=train_giu)

model.learn(total_timesteps=parameters['Total_timesteps'],callback=eval_callback,log_interval=1)

model.save(filename+'/final_model.zip')
print(filename)

with np.load(filename+'/evaluations.npz') as data:
    for j in range(data['timesteps'].shape[0]):
        print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

input("Press Enter to continue...")

if os.path.isfile(filename+'/best_model.zip'):
    path = filename+'/best_model.zip'
else:
    print("[ERROR]: no model under the specified path", filename)
model = DDPG.load(path)

eval_env.close()
train_env.close()

test_env = RLEnvironment(parameters=parameters, gui=True )

logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
            num_drones=1,
            output_folder=output_folder,
            )

mean_reward, std_reward = evaluate_policy(model,
                                            test_env,
                                            n_eval_episodes=10
                                            )
print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

obs, info = test_env.reset(seed=42, options={})
start = time.time()

for i in range((test_env.EPISODE_LEN_SEC+4)*test_env.CTRL_FREQ):
    action, _states = model.predict(obs,
                                    deterministic=True
                                    )
    obs, reward, terminated, truncated, info = test_env.step(action)
    print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
    
    test_env.render()
    print(terminated)
    sync(i, start, test_env.CTRL_TIMESTEP)
    if terminated:
        obs = test_env.reset(seed=42, options={})
test_env.close()