import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *

state_flag = 3
figObject = None

#Qlearning_model = Qlearning(state_grid_flag = 0, action_grid_flag = 0,
#  reward_flag = 2, Qfile = "Q.csv")
#Qlearning_model.train()
#figObject = Qlearning_model.test( tmax = 10, state_flag= state_flag,
#   gamma =1, figObject = figObject)


VIteration_model = ValueIteration(state_grid_flag = 0, action_grid_flag = 2,
 reward_flag = 3, Ufile = "VI8.csv")
#VIteration_model.train()
figObject = VIteration_model.test(tmax = 10, state_flag = state_flag,
  use_continuous_actions = True, gamma = 3, figObject = figObject)
#VIteration_model.heatmap_value_function()
#VIteration_model.heatmap_of_policy(option = "zero")

#I don't need to store the figObject returned by test. this returns the same
# figObject as before. We only get a new FigObject when we initialize figObject
# to None
figObject = runBicycleTest(state_flag, controller = LinearController.LinearController(),
  time = 10, isGraphing  = True, figObject = figObject)


plt.show()
plt.close("all")
