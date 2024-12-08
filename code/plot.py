import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

filepath = "results/SAC_save-12.06.2024_17.36.39"

plot = Plot(filepath)
plot.PlotReward()
plot.VisValue('final')



