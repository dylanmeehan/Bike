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

name = "VI_r14_s6_a1"
VI_model = ValueIteration(state_grid_flag = 6, action_grid_flag = 1,
 reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = True, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name, timestep = 1/50)

VI_model.train( gamma = 1, num_episodes = 30,
       interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


VI_model.init_controller(use_continuous_actions = True,
    use_continuous_state_with_discrete_actions = True,
    controller_integration_method = "fixed_step_RK4",
    use_regression_model_of_table = False)

(_, figObject) = runBicycleTest(state_flag1, VI_model, name, reward_flag = 1,
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


# name = "VI50-s11_lin"
# VIteration_model = ValueIteration(state_grid_flag = 11, action_grid_flag = 1,
#  reward_flag = 11, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
#  remake_table = False, step_table_integration_method = "fixed_step_RK4",
#  USE_LINEAR_EOM = True, name = name, timestep = 1/50)



# # VIteration_model.train( gamma = 1, num_episodes = 30,
# #        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)

# # VIteration_model.run_regression()

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, use_continuous_state_with_discrete_actions = False,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+"",
#    plot_is_inside_last_gridpoint = False, use_regression = False)



# name = "VI50-s8_lin"
# VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
#  reward_flag = 11, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
#  remake_table = False, step_table_integration_method = "fixed_step_RK4",
#  USE_LINEAR_EOM = True, name = name, timestep = 1/50)



# # # # VIteration_model.train( gamma = 1, num_episodes = 30,
# # # #        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)

# # # # # VIteration_model.run_regression()

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, use_continuous_state_with_discrete_actions = False,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+"",
#    plot_is_inside_last_gridpoint = False, use_regression = False)

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, use_continuous_state_with_discrete_actions = True,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+": cont S, A, regression",
#    plot_is_inside_last_gridpoint = False, use_regression = True)

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = False, use_continuous_state_with_discrete_actions = True,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+": cont S, dicr A",
#    plot_is_inside_last_gridpoint = False, use_regression = False)

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = False, use_continuous_state_with_discrete_actions = True,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+": cont S, dicr A, regression",
#    plot_is_inside_last_gridpoint = False, use_regression = True)





###############################################################################

#VIteration_model.heatmap_value_function("average", use_regression = False)
###############################################################################


#LQR controller with PSD Q matrix. reward flag = 11
# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 40.352939, k2 = 5.7491159,
#   k3 = -7.8522597), reward_flag = 11.1, time = simulation_duration,
#   isGraphing  = True, figObject = figObject,
#   name = "LQR, r11, Q has 1 big term", USE_LINEAR_EOM = False, timestep = 1/500)

# LQR controller with PSD Q matrix. reward flag = 11

# #LQR controller with 0's in Q matrix. reward flag = 7
# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 40.352891, k2 = 5.7490112, k3 = -7.8522343),
#   reward_flag = 7, time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "r7_LQR", USE_LINEAR_EOM = False, timestep = 1/50)


# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 39.827362, k2 = 4.5645095,k3 = -7.5558888),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "r7_LQR", USE_LINEAR_EOM = False, timestep = 1/50)


# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 8.6783612, k2 = 2.2155966,
#   k3 = -2.2155966),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "LQR with I costs", USE_LINEAR_EOM = False, timestep = 1/50)

# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 40.3528919, k2 = 5.7490112, k3 = -7.8522343),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "small timestep", USE_LINEAR_EOM = False, timestep = 1/500)


(_, figObject) = runBicycleTest(state_flag1,
  controller = LinearController.LinearController(k1 = 24, k2 = 7, k3 = -8),
  name = "working LQR, tstep = 1/50", reward_flag = 1,
  simulation_duration = simulation_duration, isGraphing  = True, figObject = figObject,
  USE_LINEAR_EOM = False , timestep = 1/50)

(_, figObject) = runBicycleTest(state_flag1,
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

