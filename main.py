import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *

state_flag1 = 3
state_flag2 = 5
figObject = None
time = 3

#Qlearning_model = Qlearning(state_grid_flag = 0, action_grid_flag = 0,
#  reward_flag = 2, Qfile = "Q.csv")
# #Qlearning_model.train()
#figObject = Qlearning_model.test( tmax = 10, state_flag= state_flag,
#   gamma =1, figObject = figObject)


# VIteration_model = ValueIteration(state_grid_flag = 1, action_grid_flag = 1,
#  reward_flag = 3, Ufile = "models/VI29.csv", use_only_continuous_actions = True)
# VIteration_model.train( gamma = 0.95, num_episodes = 100,
#       do_interpolation = True, use_continuous_actions = False)
# figObject = VIteration_model.test(tmax = time, state_flag = state_flag2,
#    use_continuous_actions = True, gamma = 1, figObject = figObject)


# VIteration_model1 = ValueIteration(state_grid_flag = 1, action_grid_flag = 1,
#  reward_flag = 3, Ufile = "models/VI26.csv", use_only_continuous_actions = True)
# # VIteration_model.train( gamma = 0.95, num_episodes = 100,
# #      do_interpolation = True, use_continuous_actions = False)
# figObject = VIteration_model1.test(tmax = time, state_flag = state_flag2,
#    use_continuous_actions = True, gamma = 1, figObject = figObject)

# VIteration_model = ValueIteration(state_grid_flag = 1, action_grid_flag = 1,
#  reward_flag = 3, Ufile = "models/VI25.csv", use_only_continuous_actions = True)
# # VIteration_model.train( gamma = 0.95, num_episodes = 100,
# #      do_interpolation = True, use_continuous_actions = False)
# figObject = VIteration_model.test(tmax = time, state_flag = state_flag1,
#   use_continuous_actions = True, gamma = 1, figObject = figObject)





#I don't need to store the figObject returned by test. this returns the same
# figObject as before. We only get a new FigObject when we initialize figObject
# to None
[success, states, figObject] = runBicycleTest(state_flag1,
  controller = LinearController.LinearController(),
  time = time, isGraphing  = True, figObject = figObject,
  tstep_multiplier = 1, name = "1Euler_LQR")
[success, states, figObject] = runBicycleTest(state_flag1,
  controller = LinearController.LinearController(),
  time = time, isGraphing  = True, figObject = figObject,
  tstep_multiplier = 10, name = "10Euler_LQR")

#VIteration_model.heatmap_value_function()
#VIteration_model.heatmap_of_policy(option = "zero", include_linear_controller = True,
#  use_continuous_actions = True )


plt.show()



plt.close("all")

