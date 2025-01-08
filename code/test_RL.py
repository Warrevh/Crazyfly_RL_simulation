import time
import numpy as np

from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy

from RLEnvironment import RLEnvironment
from data_handling import Logger_obs

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


test_env = RLEnvironment( parameters=parameters ,gui=True )

folder = "results/TD3_save-01.07.2025_16.13.26"

#model = TD3.load("results/trained big box 2.0 save-11.21.2024_23.05.24/final_model.zip")
#model = TD3.load("results/trained big box 2.0 save-11.21.2024_23.05.24/best_model.zip")
#model = TD3.load("results/trained big box save-11.20.2024_21.19.39/final_model.zip")
#model = TD3.load("results/trained big box save-11.20.2024_21.19.39/best_model.zip")
model = TD3.load(str(folder+f"/best_model"))

log = Logger_obs(folder)

mean_reward, std_reward = evaluate_policy(model,
                                            test_env,
                                            n_eval_episodes=1
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
    
    log.log_obs(obs)

    test_env.render()
    print(terminated)
    #sync(i, start, test_env.CTRL_TIMESTEP)
    if terminated:
        obs, info = test_env.reset(seed=42, options={})
        break
test_env.close()
log.save_obs()
