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
    np.meshgrid(GridPoints.phi_grid, GridPoints.phi_dot_grid, [0.0])
  phi_and_phi_dot_points= np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  n = len(controllers)
  fig1, ax1s = plt.subplots(1,n)

  difference_array = np.zeros((GridPoints.len_phi_grid, GridPoints.len_phi_dot_grid))

  for idx in range(n):
    controller = controllers[idx]

    success_array = np.zeros((GridPoints.len_phi_grid,
      GridPoints.len_phi_dot_grid))

    for (i, phi) in enumerate(GridPoints.phi_grid):
      print("phi: " + str(phi))
      for (j, phi_dot) in enumerate(GridPoints.phi_dot_grid):

        (success, _ ) = runBicycleTest(stateflag = None, controller = controller,
        name = "", reward_flag = 14, simulation_duration= 2.0,
        isGraphing  = False, figObject = None, isPrinting = False,
        integrator_method = "fixed_step_RK4",
        USE_LINEAR_EOM = False, timestep = 1/50, starting_state3 = [phi, phi_dot, 0.0])

        success_array[i,j] = success

    if idx==0:
      difference_array += success_array
    elif idx==1:
      difference_array -= success_array
    else:
      print("** Difference Array not computed for index " + str(idx))

    ax1 = ax1s[idx]

    im1 = ax1.imshow(success_array, cmap=plt.get_cmap("coolwarm"))
    ax1.set_title(str(names[idx])+" Basin of Attraction (delta = 0)")
    ax1.set_ylabel("lean [rad]")
    ax1.set_xlabel("lean rate [rad/s]")
    ax1.set_yticks(np.arange(GridPoints.len_phi_grid))
    ax1.set_xticks(np.arange(GridPoints.len_phi_dot_grid))
    ax1.set_yticklabels(np.around(GridPoints.phi_grid, decimals=2))
    ax1.set_xticklabels(np.around(GridPoints.phi_dot_grid, decimals = 2))

    fig1.colorbar(im1)

    np.savetxt("BasinOfAttraction/"+name + "_BasisOfAttraction.csv", success_array,
      delimiter = ",")

  fig2, ax2 = plt.subplots(1,1)
  #plot differences

  im2 = ax2.imshow(difference_array)
  ax2.set_title(str(names[idx])+" Difference between linear and VI controllers")
  ax2.set_ylabel("lean [rad]")
  ax2.set_xlabel("lean rate [rad/s]")
  ax2.set_yticks(np.arange(GridPoints.len_phi_grid))
  ax2.set_xticks(np.arange(GridPoints.len_phi_dot_grid))
  ax2.set_yticklabels(np.around(GridPoints.phi_grid, decimals=2))
  ax2.set_xticklabels(np.around(GridPoints.phi_dot_grid, decimals = 2))

  fig2.colorbar(im2)

  t2 = time.time()
  print("plotted basis of attraction in " + str(t2-t1) + " sec")

  plt.show()

#name = "VI_r14_s6_a1"
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

plot_basin_of_attraction([LinearController.LinearController(k1 = 27.75, k2 = 6.30,
    k3 = -8.04), VI_model],
  ["linear r14", name],  16)