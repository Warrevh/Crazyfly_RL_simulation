import numpy as np
from train_DDPG import Train_DDPG
from train_SAC import Train_SAC
from train_TD3 import Train_TD3
from train_PPO import Train_PPO

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
    'target_update_interval': 10,
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))+self.Rew_disway_fact*max(0,2-np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis)-self.Rew_angvel_fact*(np.sum((self.angvel-prev_angvel)**2)))',
    'parent_model': "none"
}

#DDPG = Train_DDPG(parameters=parameters,train_gui=False)
#TD3 = Train_TD3(parameters=parameters,train_gui=False)
#PPO = Train_PPO(parameters=parameters,train_gui=False)
SAC = Train_SAC(parameters=parameters,train_gui=False)

#DDPG.train_DDPG()
#TD3.train_TD3()
#PPO.train_PPO()
SAC.train_SAC()