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

t0 = time.time()

VIteration_model = ValueIteration(state_grid_flag = 4, action_grid_flag = 1,
 reward_flag = 3, Ufile = "models/VI33.csv", use_only_continuous_actions = False)

t1 = time.time()
print("Initialized VI Model in " + str(t1-t0) + "sec")


VIteration_model.train( gamma = 0.95, num_episodes = 10,
      interpolation_method = "linear", use_continuous_actions = False, vectorize = True)

t2 = time.time()
print("Trained VI Model in " + str(t2-t1) + "sec")

#sys.exit()

figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
   use_continuous_actions = True, gamma = 1, figObject = figObject)

t3 = time.time()
print("Tested VI Model in " + str(t3-t2))






#I don't need to store the figObject returned by test. this returns the same
# figObject as before. We only get a new FigObject when we initialize figObject
# to None
t1 = time.time()

[success, states, figObject] = runBicycleTest(state_flag1,
  controller = LinearController.LinearController(),
  time = simulation_duration, isGraphing  = True, figObject = figObject,
  name = "LQR")

t2 = time.time()
print("Tested linear controller in " + str(t2-t1) +"sec")




#VIteration_model.heatmap_value_function()
#VIteration_model.heatmap_of_policy(option = "zero", include_linear_controller = True,
#  use_continuous_actions = True )


plt.show()



plt.close("all")

