import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *
import time
import sys

state_flag1 = 6
state_flag2 = 7
figObject = None
simulation_duration = 5

###############################################################################

name = "VI_r14_a1_s16"
#name = "VI_r14_a1_s16_30episodes"
VI_model = ValueIteration(state_grid_flag = 16, action_grid_flag = 1,
reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
remake_table = False, step_table_integration_method = "fixed_step_RK4",
USE_LINEAR_EOM = False, name = name, timestep = 1/50)

# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


VI_model.init_controller(use_continuous_actions = True,
  use_continuous_state_with_discrete_actions = True,
  controller_integration_method = "fixed_step_RK4",
  use_regression_model_of_table = False)

for idx in [4,7]:

  (_, figObject) = runBicycleTest(idx, VI_model, name, reward_flag = 1,
    simulation_duration = simulation_duration, isGraphing  = True, figObject = figObject,
    integrator_method = "fixed_step_RK4",
    USE_LINEAR_EOM = False, timestep = 1/50)

# # figObject = VI_odel.test(tmax = simulation_duration, state_flag = state_flag1,
#   use_continuous_actions = True, use_continuous_state_with_discrete_actions = False,
#   gamma = 1, figObject = figObject,
#   integration_method = "fixed_step_RK4", name = name+"",
#   plot_is_inside_last_gridpoint = False, use_regression = False,
#   timesteps_to_graph_actions_vs_utilites = [14])


#######################################################################3333



# (_, figObject) = runBicycleTest(state_flag1,
#   controller = LinearController.LinearController(k1 = 24, k2 = 7, k3 = -8),
#   name = "working LQR, tstep = 1/50", reward_flag = 1,
#   simulation_duration = simulation_duration, isGraphing  = True, figObject = figObject,
#   USE_LINEAR_EOM = False , timestep = 1/50)

  (_, figObject) = runBicycleTest(idx,
    controller = LinearController.LinearController(k1 = 27.75, k2 = 6.30,
    k3 = -8.04), name = "LQR, r14, 1/100", reward_flag = 1,
    simulation_duration = simulation_duration,
    isGraphing  = True, figObject = figObject,
    USE_LINEAR_EOM = False, timestep = 1/100)

# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 43.817, k2 = 12.18,
#   k3 = -9.26), reward_flag = 11.1, time = simulation_duration,
#   isGraphing  = True, figObject = figObject,
#   name = "LQR, r12, 1/50", USE_LINEAR_EOM = False, timestep = 1/50)

# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 43.817, k2 = 12.18,
#   k3 = -9.26), reward_flag = 11.1, time = simulation_duration,
#   isGraphing  = True, figObject = figObject,
#   name = "LQR, r12, 1/70", USE_LINEAR_EOM = False, timestep = 1/70)


# t2 = time.time()



plt.show()



plt.close("all")

