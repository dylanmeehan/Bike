import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *
from LinearController import getLQRGains
import time
import sys

state_flag1 = 6
state_flag2 = 7
figObject = None
simulation_duration = 5
make_graph = True

###############################################################################

# name = "VI_r14_s16_a1_30episodes"
# name = "VI_r14_a1_s16_v1_50episodes"
v = 0.5
name = "VI_r14_a1_s16_v0.5_100episodes"
VI_model = ValueIteration(state_grid_flag = 16, action_grid_flag = 1,
reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
remake_table = False, step_table_integration_method = "fixed_step_RK4",
USE_LINEAR_EOM = False, name = name, timestep = 1/50, v = v)

# VI_model.train( gamma = 1, num_episodes = 300, convergence_threshold = 0.9995,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)

VI_model.init_controller(use_continuous_actions = True,
  use_continuous_state_with_discrete_actions = True,
  controller_integration_method = "fixed_step_RK4",
  use_regression_model_of_table = False)

for starting_state in [93,94,95]:
#######################################################################3333

  (_, figObject) = runBicycleTest(starting_state, VI_model, name = "VI, n",
     reward_flag = 1,
    simulation_duration = simulation_duration, isGraphing  = make_graph,
    figObject = figObject,
    integrator_method = "fixed_step_RK4",
    USE_LINEAR_EOM = False, timestep = 1/50, v = v)

  # LQR_gains = getLQRGains("lqrd_0.5m_s")
  # (_, figObject) = runBicycleTest(starting_state,
  #   controller = LinearController.LinearController(LQR_gains),
  #   name = "lqrd_0.5m_s", reward_flag = 14,
  #   simulation_duration = simulation_duration,
  #   isGraphing  = make_graph, figObject = figObject,
  #   USE_LINEAR_EOM = False, timestep = 1/50, v = v)

  LQR_gains = getLQRGains("lqrd_0.5m_s")
  (_, figObject) = runBicycleTest(starting_state,
    controller = LinearController.LinearController(LQR_gains),
    name = "lqrd_0.5m_s", reward_flag = 14,
    simulation_duration = simulation_duration,
    isGraphing  = make_graph, figObject = figObject,
    USE_LINEAR_EOM = False, timestep = 1/50, v = v)



  plt.show()



plt.close("all")

