import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from datetime import datetime
import csv
import pickle

from stable_baselines3 import TD3


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

        model = TD3.load(filename)

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
        plt.colorbar(label="Average Value")
        plt.xlabel("X (State)")
        plt.ylabel("Y (State)")
        plt.title("Average Value Contour Map")

        target_x = 2.5
        target_y = 2
        plt.scatter(target_x, target_y, color='red', s=50, label="Target")
        plt.legend()

        plt.savefig(self.plot_path+"/"+model_type+"_average_value_contour.png", dpi=1000)
        plt.show()

    def PlotReward(self):
        filename = os.path.join(self.file, 'evaluations.npz')

        data = np.load(filename)

        steps = data['timesteps'] 
        rewards = data['results']

        avg_rewards = rewards.mean(axis=1)

        window = 1
        moving_avg_rewards = np.convolve(avg_rewards, np.ones(window)/window, mode='valid')

        steps_moving_avg = steps[:len(moving_avg_rewards)]


        print(np.max(avg_rewards))

        plt.figure(figsize=(10, 6))
        plt.plot(steps_moving_avg, moving_avg_rewards, label="MA Return vs Steps", color="blue")

        plt.title("MA Return vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_path+"/Return.png", dpi=1000)
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
    
class Logger_obs():
    def __init__(self,store_path):
        self.store_path = store_path
        current_time = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        self.file_path = str(self.store_path + f"/obs_log_sim_{current_time}.txt")

        self.all_obs = []

    def log_obs(self,obs):
        flattened_obs = [
            obs["Position"][0][0], obs["Position"][0][1], obs["Position"][0][2],  # Position (x, y, z)
            obs["Velocity"][0][0], obs["Velocity"][0][1], obs["Velocity"][0][2],  # Velocity (x, y, z)
            obs["rpy"][0][0], obs["rpy"][0][1], obs["rpy"][0][2],                # Roll, Pitch, Yaw
            obs["ang_v"][0][0], obs["ang_v"][0][1], obs["ang_v"][0][2]          # Angular velocity (x, y, z)
        ]
        self.all_obs.append(flattened_obs)  # Append the observation to the log

    def save_obs(self):
        headers = [
            "Position_x", "Position_y", "Position_z",
            "Velocity_x", "Velocity_y", "Velocity_z",
            "Roll", "Pitch", "Yaw",
            "AngularVelocity_x", "AngularVelocity_y", "AngularVelocity_z"
        ]
        
        # Write the collected data to a CSV file
        with open(self.file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write the header row
            writer.writerows(self.all_obs)  # Write all the collected data
            print("observation written to file "+ self.file_path)

class Plot_obs():
    def __init__(self,file):
        self.file = file
        self.all_obs = np.loadtxt(self.file,delimiter=',', skiprows=1)

        self.column_indices = {
                "Position_x": 0, "Position_y": 1, "Position_z": 2,        # Position
                "Velocity_x": 3, "Velocity_y": 4, "Velocity_z": 5,     # Velocity
                "Roll": 6, "Pitch": 7, "Yaw": 8,        # Roll, Pitch, Yaw
                "AngularVelocity_x": 9, "AngularVelocity_y": 10, "AngularVelocity_z": 11    # Angular Velocity
            }
        
        self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    def plot_single_value(self,value):
        value_index = self.get_column_index(value)
        plotted_value = [obs[value_index] for obs in self.all_obs]
        steps = list(range(1, len(self.all_obs) + 1)) 

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(steps, plotted_value, linestyle='-', color='blue', label= value)

        # Add labels and title
        plt.xlabel('Steps')
        plt.ylabel(value)
        plt.title(value+' vs. Steps')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_multiple_value(self,values):
        steps = list(range(1, len(self.all_obs) + 1))
        plt.figure(figsize=(8, 5))
        for i, value in enumerate(values):
            value_index = self.get_column_index(value)
            plotted_value = [obs[value_index] for obs in self.all_obs]
            color = self.colors[i % len(self.colors)]
            plt.plot(steps, plotted_value, linestyle='-', color=color, label= value)


        plt.xlabel('Steps')
        plt.ylabel('m')
        plt.title('Position vs. Steps')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
            
    def plot_xy_position(self):
        x_pos = [obs[0] for obs in self.all_obs]
        y_pos = [obs[1] for obs in self.all_obs]

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(x_pos, y_pos, linestyle='-', color='blue', label= 'position')

        # Add labels and title
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title('Drone path')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    # Function to get the corresponding column index
    def get_column_index(self,name):
        return self.column_indices.get(name, "Invalid name")
    
class Plot_muliple_runs():
    def __init__(self,file):
        self.file = file

        with open(self.file, 'rb') as f:
            self.data_all_runs = pickle.load(f)

        self.number_of_runs = len(self.data_all_runs)

    def x_y_pos_one_run(self,arr):
        pos_x_one_run = []
        pos_y_one_run = []
        for d in arr:
            obs_array = d["obs"]
            pos_x = obs_array[0]
            pos_y = obs_array[1]
            pos_x_one_run.append(pos_x)
            pos_y_one_run.append(pos_y)

        return pos_x_one_run,pos_y_one_run

    def best_reward_arr(self):
        best_array = None
        best_reward = float('-inf')

        for arr in self.data_all_runs:
            total_reward_ep = sum(d["reward"] for d in arr)
            
            if total_reward_ep > best_reward:
                best_reward = total_reward_ep
                best_array = arr
        
        return best_array
    
    def shortest_arr(self):
        shortest_array = None
        shortest_length = float('inf')
        
        for arr in self.data_all_runs:
            array_length = len(arr)
            
            if array_length < shortest_length:
                shortest_length = array_length
                shortest_array = arr
        return shortest_array
    
    def plot_xy_positions(self):
        plt.figure(figsize=(8, 5))

        for arr in self.data_all_runs:
            pos_x,pos_y = self.x_y_pos_one_run(arr)
            plt.plot(pos_x, pos_y, linestyle='-', color='red', alpha=0.1)

        pos_x,pos_y = self.x_y_pos_one_run(self.best_reward_arr())
        plt.plot(pos_x, pos_y, linestyle='-', color='green', alpha=0.9, label= "Best reward" )

        # Add labels and title
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.title(f'Path of {self.number_of_runs} runs ')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_distribution_return(self):
        return_ep = []

        for arr in self.data_all_runs:
            return_ep.append(sum(d["reward"] for d in arr))

        plt.figure(figsize=(8, 6))
        sns.histplot(return_ep, kde=True, bins=50, color='blue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Return', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Return for {self.number_of_runs} runs', fontsize=14)

        # Show the plot
        plt.show()


