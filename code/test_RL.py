import time
import numpy as np

from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import TD3,SAC
from stable_baselines3.common.evaluation import evaluate_policy

from RLEnvironment import RLEnvironment
from data_handling import Logger_obs

#used for testing the models

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


test_env = RLEnvironment( parameters=parameters ,gui=True )

#model to test
folder = "results/SAC_save-01.15.2025_00.49.25"

#type of model: best or final
model = SAC.load(str(folder+f"/best_model"))

log = Logger_obs(folder)

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
    
    log.log_obs(obs)

    test_env.render()
    print(terminated)
    #sync(i, start, test_env.CTRL_TIMESTEP)
    if terminated:
        obs, info = test_env.reset(seed=42, options={})
        break
test_env.close()
log.save_obs()
