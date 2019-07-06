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
for i in range(4):
  v = [1.0, 2.0, 2.5, 3.0][i]
  name = ["VI_r14_a8_s16_v1_g95_limB_300episodes",
          "VI_r14_a8_s16_v2_g95_limB_300episodes",
          "VI_r14_a8_s16_v2_5_g95_limB_300episodes",
          "VI_r14_a8_s16_v3_g95_limB_300episodes"][i]
  VI_model = ValueIteration(state_grid_flag = 16, action_grid_flag = 8,
  reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
  remake_table = False, step_table_integration_method = "fixed_step_RK4",
  USE_LINEAR_EOM = False, name = name, timestep = 1/50, v = v)

  VI_model.train( gamma = 0.95, num_episodes = 300, value_convergence_threshold = 0.99,
         interpolation_method = "linear", use_continuous_actions = False, vectorize = None)

  VI_model.init_controller(use_continuous_actions = True,
    use_continuous_state_with_discrete_actions = True,
    controller_integration_method = "fixed_step_RK4",
    use_regression_model_of_table = False)

figObject = None
#starting_state3 = [-0.6,-0.8, 0.0]
starting_state3 = [-.2, -.1, 0.0]


LQR_gains = getLQRGains("lqrd_1m_s")
(_, figObject) = runBicycleTest(None,
controller = LinearController.LinearController(LQR_gains),
name = "LQR", reward_flag = 14,
simulation_duration = simulation_duration,
isGraphing  = make_graph, figObject = figObject, starting_state3 = starting_state3,
USE_LINEAR_EOM = False, timestep = 1/50, v = v)


# LQR_gains = getLQRGains("lqrd_2m_s")
# (_, figObject) = runBicycleTest(None,
# controller = LinearController.LinearController(LQR_gains),
# name = "steer angle limit 0.7, rate limit 1 rad/s", reward_flag = 14,
# simulation_duration = simulation_duration,
# isGraphing  = make_graph, figObject = figObject, starting_state3 = starting_state3,
# USE_LINEAR_EOM = False, timestep = 1/50, v = v)

(_, figObject) = runBicycleTest(None,
VI_model,
name = "VI", reward_flag = 14,
simulation_duration = simulation_duration,
isGraphing  = make_graph, figObject = figObject, starting_state3 = starting_state3,
USE_LINEAR_EOM = False, timestep = 1/50, v = v)




plt.show()



plt.close("all")


