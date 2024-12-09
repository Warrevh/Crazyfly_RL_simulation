import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from stable_baselines3 import DDPG,SAC


class Txt_File:
    def __init__(self,store_path):
        self.store_path = store_path

        self.title = 'PARAMETERS'

    def save_parameters(self, par):
        file_path = self.store_path + f"/parameters.txt"

        with open(file_path, 'w') as f:
            f.write(f"{self.title}\n\n\n")
            for key, value in par.items():
                f.write(f"{key}: {value}\n")

class Plot:
    def __init__(self,file):
        self.file = file

        self.plot_path = os.path.join(file, 'plots')
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path+'/')

    def VisValue(self,model_type):

        model_type_ = model_type + '_model.zip'
        filename = os.path.join(self.file, model_type_)

        model = DDPG.load(filename)

        critic = model.policy.critic
        actor = model.policy.actor

        x_range = np.linspace(0, 5, 500)
        y_range = np.linspace(0, 4, 400)

        """
        action = np.array([[-1,0]])
        action_tensor = torch.tensor(action)
        """


        q_values = np.zeros((len(x_range), len(y_range)))

        for i, x in enumerate(x_range):
            print(i)
            for j, y in enumerate(y_range):
                state = self.dictState(x, y)
                action_predicted = actor(state)
                with model.policy.device:
                    q_value = critic(state, action_predicted)[0].item()
                q_values[i, j] = q_value

        plt.figure(figsize=(8, 6))
        plt.contourf(x_range, y_range, q_values.T, levels=50, cmap='viridis')
        plt.colorbar(label="Average Q-Value")
        plt.xlabel("X (State)")
        plt.ylabel("Y (State)")
        plt.title("Average Q-Value Contour Map")

        target_x = 2.5
        target_y = 2
        plt.scatter(target_x, target_y, color='red', s=50, label="Target")
        plt.legend()

        plt.savefig(self.plot_path+"/"+model_type+"_average_q_value_contour.png", dpi=1000)
        plt.show()

    def PlotReward(self):
        filename = os.path.join(self.file, 'evaluations.npz')

        data = np.load(filename)

        steps = data['timesteps'] 
        rewards = data['results']

        avg_rewards = rewards.mean(axis=1)

        print(np.max(avg_rewards))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, avg_rewards, label="Reward vs Steps", color="blue")

        plt.title("Reward vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_path+"/Rewards.png", dpi=1000)
        plt.show()

    def dictState(self,x,y):
        NUM_DRONES = 1
        vel = np.zeros((NUM_DRONES,3))
        rpy = np.zeros((NUM_DRONES,3))
        ang_v = np.zeros((NUM_DRONES,3))
    
        pos = np.array([[x,y,0.2]])

        ret = {
                "Position": torch.tensor(np.array([pos[i,:] for i in range(NUM_DRONES)]).astype('float32')),
                "Velocity": torch.tensor(np.array([vel[i,:] for i in range(NUM_DRONES)]).astype('float32')),
                "rpy": torch.tensor(np.array([rpy[i,:] for i in range(NUM_DRONES)]).astype('float32')),
                "ang_v": torch.tensor(np.array([ang_v[i,:] for i in range(NUM_DRONES)]).astype('float32')),
            }
        
        return ret