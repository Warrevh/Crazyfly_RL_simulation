import time
import numpy as np

from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.evaluation import evaluate_policy

from RLEnvironment import RLEnvironment

parameters = {
    #env parameters
    'initial_xyzs': np.array([[4.5,3.5,0.2]]),
    'random_initial_pos': True,
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
    'Total_timesteps': int(10e6),
    'train_freq': 1,
    'Reward_Function': '(-self.Rew_distrav_fact*(np.linalg.norm(self.reward_state[0:2]-prev_state[0:2]))-self.Rew_disway_fact*(np.linalg.norm(self.TARGET_POS[0:2]-self.reward_state[0:2])**4)-self.Rew_step_fact*1 +self.Rew_tardis_fact*(prev_tar_dis-self.target_dis))'
}

start_pos= np.array([[1,3.5,0.2]]) #np.array([[4.5,3.5,0.2]])  # 

test_env = RLEnvironment( parameters=parameters ,gui=True )

#model = DDPG.load("results/trained big box 2.0 save-11.21.2024_23.05.24/final_model.zip")
#model = DDPG.load("results/trained big box 2.0 save-11.21.2024_23.05.24/best_model.zip")
#model = DDPG.load("results/trained big box save-11.20.2024_21.19.39/final_model.zip")
#model = DDPG.load("results/trained big box save-11.20.2024_21.19.39/best_model.zip")
model = SAC.load("results/SAC_save-12.07.2024_22.03.53/best_model.zip")

mean_reward, std_reward = evaluate_policy(model,
                                            test_env,
                                            n_eval_episodes=5
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