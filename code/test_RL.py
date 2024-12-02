import time
import numpy as np

from gym_pybullet_drones.utils.utils import sync

from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

from RLEnvironment import RLEnvironment

parameters = {
    #env parameters
    'initial_xyzs': np.array([[1,3.5,0.2]]),
    'ctrl_freq': 240,
    'Target_pos': np.array([2.5,2,0.2]),
    'episode_length': 60,
    #Learning rate
    'Learning_rate': 0.0005,
    'Learning_rate_decay': -0.005,
    #Reward
    'Target_reward': -700,
    #Reward Function
    'Rew_distrav_factor': 0.1,
    'Rew_disway_factor': 0.1,
    #evaluation callback
    'eval_freq': 5, #"epsisodes" (eval_freq*(epsiode_length*ctrl_freq))
    #observation !!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'position': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'velocity': True, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'rpy': False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'ang_v': False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    'prev_act':False, #!!!!!!! ADJUST MANUALY IN CODE !!!!!!!
    #train
    'Total_timesteps': int(10e6),
}

start_pos= np.array([[1,3.5,0.2]]) #np.array([[4.5,3.5,0.2]])  # 

test_env = RLEnvironment( parameters=parameters ,gui=True )

#model = DDPG.load("results/trained big box 2.0 save-11.21.2024_23.05.24/final_model.zip")
#model = DDPG.load("results/trained big box 2.0 save-11.21.2024_23.05.24/best_model.zip")
#model = DDPG.load("results/trained big box save-11.20.2024_21.19.39/final_model.zip")
#model = DDPG.load("results/trained big box save-11.20.2024_21.19.39/best_model.zip")
model = DDPG.load("results/save-11.30.2024_14.36.35/best_model.zip")

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