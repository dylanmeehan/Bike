import LinearController
from StateGridPoints import *
from runBicycleTest import *
import time
from valueIteration import *


def plot_basin_of_attraction(controllers, names, state_table_flag):
  t1 = time.time()

  GridPoints = StateGridPoints()
  GridPoints.set_state_grid_points(state_table_flag)

  #with delta = 0

  phi_points, phi_dot_points,  delta_points = \
    np.meshgrid(GridPoints.phi_grid, GridPoints.phi_dot_grid, [0])
  phi_and_phi_dot_points= np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  n = len(controllers)
  fig1, ax1s = plt.subplots(1,n)

  for idx in range(n):
    controller = controllers[idx]

    success_array = np.zeros((GridPoints.len_phi_grid,
      GridPoints.len_phi_dot_grid))

    for (i, phi) in enumerate(GridPoints.phi_grid):
      for (j, phi_dot) in enumerate(GridPoints.phi_dot_grid):

        (success, _ ) = runBicycleTest(stateflag = None, controller = controller,
        name = "", reward_flag = 0, simulation_duration= 4,
        isGraphing  = False, figObject = None,
        integrator_method = "fixed_step_RK4",
        USE_LINEAR_EOM = False, timestep = 1/50, starting_state3 = [phi, phi_dot, 0])

        success_array[i,j] = success

    ax1 = ax1s[idx]

    im1 = ax1.imshow(success_array, cmap=plt.get_cmap("coolwarm"))
    ax1.set_title(str(names[idx])+" Basin of Attraction (delta = 0)")
    ax1.set_ylabel("lean [rad]")
    ax1.set_xlabel("lean rate [rad/s]")
    ax1.set_yticks(np.arange(GridPoints.len_phi_grid))
    ax1.set_xticks(np.arange(GridPoints.len_phi_dot_grid))
    ax1.set_yticklabels(GridPoints.phi_grid)
    ax1.set_xticklabels(GridPoints.phi_dot_grid)

    fig1.colorbar(im1)

  t2 = time.time()
  print("plotted basis of attraction in " + str(t2-t1) + " sec")

  plt.show()


name = "VI_r14_a1_s16_30episodes"
VI_model = ValueIteration(state_grid_flag = 16, action_grid_flag = 1,
reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
remake_table = False, step_table_integration_method = "fixed_step_RK4",
USE_LINEAR_EOM = False, name = name, timestep = 1/50)

# VIteration_model.train( gamma = 1, num_episodes = 30,
#        interpolation_method = "linear", use_continuous_actions = False, vectorize = None)


VI_model.init_controller(use_continuous_actions = True,
  use_continuous_state_with_discrete_actions = True,
  controller_integration_method = "fixed_step_RK4",
  use_regression_model_of_table = False)

plot_basin_of_attraction([LinearController.LinearController(), VI_model],
  ["linear", name],  16.1)