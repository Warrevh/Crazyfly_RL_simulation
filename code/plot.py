import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot,Plot_obs

model_filepath = "results\TD3_save-12.30.2024_13.54.04"

plot = Plot(model_filepath)
plot.PlotReward()
plot.VisValue('best')
"""
obs_file = 'results/SAC_save-12.21.2024_08.16.17/obs_log_sim_12.22.2024_15.29.31.txt'#'results/trained_SAC_save-12.16.2024_01.10.16/obs_log_sim_12.16.2024_15.00.56.txt'#'test_data/obs_log_filterd_2024-12-16 21:51:14.txt'
plot_obs = Plot_obs(obs_file)
#plot_obs.plot_single_value('Roll')
plot_obs.plot_multiple_value(["AngularVelocity_x","AngularVelocity_y","AngularVelocity_z"]) #"AngularVelocity_x","AngularVelocity_y","AngularVelocity_z","Roll","Pitch","Yaw"
plot_obs.plot_xy_position()
"""

