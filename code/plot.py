import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

filepath = "results/SAC_save-12.09.2024_19.47.21"

plot = Plot(filepath)
plot.PlotReward()
plot.VisValue('final')



