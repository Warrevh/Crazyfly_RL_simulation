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

class Train_DDPG():
    def __init__(self,parameters,train_gui):
        self.parameters = parameters

        self.eval_freq = self.parameters['eval_freq']*self.parameters['ctrl_freq']*self.parameters['episode_length']

        self.train_giu = train_gui

        self.output_folder= 'results'

        self.filename = os.path.join(self.output_folder, 'DDPG_save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(self.filename):
            os.makedirs(self.filename+'/')

        Myparameters = Txt_File(self.filename)
        Myparameters.save_parameters(self.parameters)

    def train_DDPG(self):

        train_env = make_vec_env(RLEnvironment,
                                 env_kwargs=dict(parameters=self.parameters),
                                 n_envs=self.parameters['number_of_env'], seed=0)

        eval_env = RLEnvironment(parameters=self.parameters,gui = self.train_giu)

        print('[INFO] Action space:', train_env.action_space)
        print('[INFO] Observation space:', train_env.observation_space)

        n_actions = train_env.action_space.shape
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1/3 * np.ones(n_actions))

        #for training from an older model
        """
        model = DDPG.load("results/DDPG_save-12.08.2024_23.19.52/final_model.zip",train_env)
        """

        #for training from new model
        model = DDPG('MultiInputPolicy',train_env,
                    learning_rate=self.parameters['Learning_rate'],
                    learning_starts=self.parameters['learning_starts'],
                    batch_size=self.parameters['batch_size'],
                    action_noise=action_noise,
                    train_freq= (int(self.parameters['train_freq']), "step"), #int(eval_env.CTRL_FREQ//2)
                    gradient_steps=self.parameters['gradient_steps'],
                    verbose=1)
    

        target_reward = self.parameters['Target_reward']

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                            verbose=1)
        eval_callback = EvalCallback(eval_env,
                                        callback_on_new_best=callback_on_best,
                                        verbose=1,
                                        n_eval_episodes= self.parameters['eval_episodes'],
                                        best_model_save_path=self.filename+'/',
                                        log_path=self.filename+'/',
                                        eval_freq=self.eval_freq,
                                        deterministic=True,
                                        render=self.train_giu)

        model.learn(total_timesteps=self.parameters['Total_timesteps'],callback=eval_callback,log_interval=self.parameters['number_of_env'])

        model.save(self.filename+'/final_model.zip')
        print(self.filename)

        with np.load(self.filename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

        input("Press Enter to continue...")

        if os.path.isfile(self.filename+'/best_model.zip'):
            path = self.filename+'/best_model.zip'
        else:
            print("[ERROR]: no model under the specified path", self.filename)
        model = DDPG.load(path)

        eval_env.close()
        train_env.close()

        test_env = RLEnvironment(parameters=self.parameters, gui=True )

        logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=self.output_folder,
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