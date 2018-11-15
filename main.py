import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *

state_flag = 3

Qlearning_model = Qlearning(state_grid_flag = 0, action_grid_flag = 0)
#Qlearning_model.train()
figObject = Qlearning_model.test(Qfile = "Q.csv", tmax = 10, state_flag= state_flag,
  gamma =1, figObject = None)


VIteration_model = ValueIteration(state_grid_flag = 0, action_grid_flag = 0)
# VIteration_model.train()
figObject2 = VIteration_model.test(Ufile = "valueIteration_U.csv", tmax = 10, state_flag = 1,
       gamma = 3, figObject = figObject)

#I don't need to store the figObject returned by test. this returns the same
# figObject as before. We only get a new FigObject when we initialize figObject
# to None
figObject3 = runBicycleTest(state_flag, controller = LinearController.LinearController(),
  time = 10, isGraphing  = True, figObject = figObject)


plt.show()
plt.close("all")
