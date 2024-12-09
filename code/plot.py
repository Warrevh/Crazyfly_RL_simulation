import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

filepath = "results/DDPG_save-12.08.2024_23.19.52"

plot = Plot(filepath)
plot.PlotReward()
plot.VisValue('best')



