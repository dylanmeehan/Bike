import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *
import time
import sys

state_flag1 = 3
state_flag2 = 5
figObject = None
simulation_duration = 2

#####################################################3


name = "VI43-s8_linear"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 3, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name)


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")

###############################################################################


name = "VI43-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 3, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name)


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")

###############################################################################


name = "VI46-s8_linear"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 7, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name)


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")

###############################################################################


name = "VI46-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 7, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name)


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")



###############################################################################
###############################################################################


# #I don't need to store the figObject returned by test. this returns the same
# # figObject as before. We only get a new FigObject when we initialize figObject
# # to None
# t1 = time.time()

# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "classic linear", USE_LINEAR_EOM = False)

[success, states, figObject] = runBicycleTest(state_flag2,
  controller = LinearController.LinearController(k1 = 40.353, k2 = 5.749, k3 = -7.852),
  time = simulation_duration, isGraphing  = True, figObject = figObject,
  name = "r7_LQR", USE_LINEAR_EOM = False)

# t2 = time.time()



plt.show()



plt.close("all")

