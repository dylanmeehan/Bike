import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *
import time
import sys

state_flag1 = 3
state_flag2 = 5
figObject = None
simulation_duration = 5

#####################################################3


VIteration_model = ValueIteration(state_grid_flag = 4, action_grid_flag = 1,
 reward_flag = 3, Ufile = "models/VI34", use_only_continuous_actions = False,
 step_table_integration_method = "fixed_step_RK4")


# VIteration_model.train( gamma = 0.95, num_episodes = 30,
#       interpolation_method = "linear", use_continuous_actions = False, vectorize = True)


figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = "VI34: continuous")

figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = False, gamma = 1, figObject = figObject,
   integration_method = "fixed_step_RK4", name = "VI34: discrete")


# VIteration_model = ValueIteration(state_grid_flag = 4, action_grid_flag = 2,
#  reward_flag = 3, Ufile = "models/VI35", use_only_continuous_actions = False,
#  step_table_integration_method = "fixed_step_RK4")

# VIteration_model.train( gamma = 0.95, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = True)


# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = "VI35: continuous")

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = False, gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = "VI35: discrete")




# #I don't need to store the figObject returned by test. this returns the same
# # figObject as before. We only get a new FigObject when we initialize figObject
# # to None
# t1 = time.time()

[success, states, figObject] = runBicycleTest(state_flag1,
  controller = LinearController.LinearController(),
  time = simulation_duration, isGraphing  = True, figObject = figObject,
  name = "LQR")

# t2 = time.time()
# print("Tested linear controller in " + str(t2-t1) +"sec")



plt.show()



plt.close("all")

