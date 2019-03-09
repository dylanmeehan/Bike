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


name = "VI43-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 3, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4")


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")



# name = "VI45-s8"
# VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
#  reward_flag = 6, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
#  remake_table = False, step_table_integration_method = "fixed_step_RK4")


# # # VIteration_model.train( gamma = 1, num_episodes = 30,
# # #        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+": continuous")

name = "VI46-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 7, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4")


# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")

name = "VI48-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 9, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4")


VIteration_model.train( gamma = 1, num_episodes = 30,
       interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = name+": continuous")


# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = False, gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+" discrete")




# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = False, gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = "VI34: discrete")


# #I don't need to store the figObject returned by test. this returns the same
# # figObject as before. We only get a new FigObject when we initialize figObject
# # to None
# t1 = time.time()

[success, states, figObject] = runBicycleTest(state_flag2,
  controller = LinearController.LinearController(),
  time = simulation_duration, isGraphing  = True, figObject = figObject,
  name = "LQR")

# t2 = time.time()



plt.show()



plt.close("all")

