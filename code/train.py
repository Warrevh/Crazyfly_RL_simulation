import numpy as np
from train_DDPG import Train_DDPG
from train_SAC import Train_SAC

parameters = {
    #env parameters
    'initial_xyzs': np.array([[2.5,3.5,0.2]]),
    'random_initial_pos': True,
    'ctrl_freq': 240,
    'Target_pos': np.array([2.5,2,0.2]),
    'episode_length': 30,
    #Learning
    'Learning_rate': 0.0001,
    'learning_starts': 100000,
    'batch_size':1000,
    'use_sde':True ,
    #Reward
    'Target_reward': 1500,
    #Reward Function
    'Rew_distrav_fact': 0,
    'Rew_disway_fact': 0.01,
    'Rew_step_fact': 0,
    'Rew_tardis_fact': 100,
    'Rew_collision': -1000,
    'Rew_terminated': 1000,
    #evaluation callback
    'eval_freq': 1, #"epsisodes" (eval_freq*(epsiode_length*ctrl_freq))
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
    'parent_model': "results/SAC_save-12.24.2024_00.37.24/final_model.zip"
}

DDPG = Train_DDPG(parameters=parameters,train_gui=False)
#SAC = Train_SAC(parameters=parameters,train_gui=False)

DDPG.train_DDPG()
#SAC.train_SAC()