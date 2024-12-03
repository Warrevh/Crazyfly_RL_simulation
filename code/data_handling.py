import numpy as np
import matplotlib.pyplot as plt


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

class plot:
    def __init__(self,model):
        self.model = model

    def VisValue(self):

        critic = self.model.policy.critic
        actor = self.model.policy.actor

        x = np.linspace(0, 5, 5000)
        y = np.linspace(0, 4, 4000)

        action_range = np.linspace(-1, 1, 1000)

        q_values = np.zeros((len(x), len(y)))

        for i, x in enumerate(x):
            for j, y in enumerate(y):
                state_tensor = self.model.policy.convert_to_tensor([x, y])
                
                action_qs = []
                for action in action_range:
                    action_tensor = self.model.policy.convert_to_tensor([[action]])
                    with self.model.policy.device:
                        q_value = critic(state_tensor, action_tensor).item()
                    action_qs.append(q_value)
                
                q_values[i, j] = np.mean(action_qs)
        

        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, q_values.T, levels=50, cmap='viridis')
        plt.colorbar(label="Average Q-Value")
        plt.xlabel("X (State)")
        plt.ylabel("Y (State)")
        plt.title("Average Q-Value Contour Map")
        plt.show()