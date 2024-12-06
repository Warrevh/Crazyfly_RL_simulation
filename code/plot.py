import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot

filepath = "results/trained_save-12.06.2024_00.43.11"

plot = Plot(filepath)
plot.PlotReward()
plot.VisValue('best')



