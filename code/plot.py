import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

filepath = "results/trained_save-12.03.2024_22.52.31"

plot = Plot(filepath)
#plot.PlotReward()
plot.VisValue('final')



