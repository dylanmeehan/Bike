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




###############################################################################

name = "VI46-s8"
VIteration_model = ValueIteration(state_grid_flag = 8, action_grid_flag = 1,
 reward_flag = 7, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
 remake_table = False, step_table_integration_method = "fixed_step_RK4",
 USE_LINEAR_EOM = False, name = name)



# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)

# VIteration_model.run_regression()

# figObject = VIteration_model.test(tmax = simulation_duration, state_flag = state_flag2,
#    use_continuous_actions = True, use_continuous_state_with_discrete_actions = True,
#     gamma = 1, figObject = figObject,
#    integration_method = "fixed_step_RK4", name = name+": cont S, A",
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
###############################################################################

VIteration_model.heatmap_value_function(option = "zero", use_regression = False)

# VIteration_model.heatmap_of_policy(option = "zero", include_linear_controller = True,
#     use_continuous_actions = False,
#     linear_controller = LinearController.LinearController(k1 = 40.353, k2 = 5.749, k3 = -7.852))





###############################################################################
###############################################################################



# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(k1 = 40.353, k2 = 5.749, k3 = -7.852),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "r7_LQR", USE_LINEAR_EOM = False)

# [success, states, figObject] = runBicycleTest(state_flag2,
#   controller = LinearController.LinearController(),
#   time = simulation_duration, isGraphing  = True, figObject = figObject,
#   name = "other_LQR", USE_LINEAR_EOM = False)

# t2 = time.time()



plt.show()



plt.close("all")

