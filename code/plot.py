import numpy as np
import matplotlib.pyplot as plt

filepath = "results/trained close target save-11.14.2024_21.36.56/evaluations.npz"

data = np.load(filepath)

print(data.keys())

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

plt.show()