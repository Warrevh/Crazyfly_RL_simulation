import numpy as np
import matplotlib.pyplot as plt

from data_handling import Plot,Plot_obs,Plot_muliple_runs

#used for visualising data

####
#plot for RL model
model_filepath = "results/SAC"

plot = Plot(model_filepath)
plot.PlotReward()
plot.VisValue('best')

####
"""
####
#plot for observations of single episode
obs_file = 'implementation_test_data/obs_log_filterd_2025-01-15 15:43:09.txt'#'results/trained_SAC_save-12.16.2024_01.10.16/obs_log_sim_12.16.2024_15.00.56.txt'#'implementation_test_data/obs_log_filterd_2024-12-16 21:51:14.txt'
obs_files = ['implementation_test_data/obs_log_filterd_2025-01-15 15:55:52.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:53:25.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:52:33.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:51:50.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:50:55.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:49:50.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:49:05.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:48:22.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:47:09.txt',
             'implementation_test_data/obs_log_filterd_2025-01-15 15:43:09.txt'
             ]
plot_obs = Plot_obs(obs_file)
#plot_obs.plot_single_value('Roll')
plot_obs.plot_multiple_value(["Velocity_x","Velocity_y"]) #"AngularVelocity_x","AngularVelocity_y","AngularVelocity_z","Roll","Pitch","Yaw"
plot_obs.plot_xy_position()
plot_obs.plot_multi_xy_position(obs_files)
####

####
#plot for plotting multiple runs
file1 = "results/DDPG/data_100_runs_with_best_model_01.16.2025_22.01.57.pkl"
file2 = "results/TD3/data_100_runs_with_best_model_01.16.2025_21.26.24.pkl"
file3 = "results/SAC/data_100_runs_with_best_model_01.16.2025_22.33.40.pkl"
file4 = "results/TD3_with_200/data_100_runs_with_best_model_01.16.2025_22.39.14.pkl"

plot_multiple = Plot_muliple_runs(file1,file2,file3,file4)
plot_multiple.plot_xy_positions()
#plot_multiple.plot_distribution_return()
#plot_multiple.plot_distribution_endpoint()
#plot_multiple.plot_boxplot_return()
#plot_multiple.plot_boxplot_endpoint()
####
"""