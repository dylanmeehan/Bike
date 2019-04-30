import LinearController
from StateGridPoints import *
from runBicycleTest import *
import time
from valueIteration import *
from LinearController import getLQRGains
from datetime import datetime


def plot_basin_of_attraction(controllers, names, state_table_flag, v):
  t1 = datetime.now()

  GridPoints = StateGridPoints()
  GridPoints.set_state_grid_points(state_table_flag)

  #with delta = 0

  phi_points, phi_dot_points,  delta_points = \
    np.meshgrid(GridPoints.phi_grid, GridPoints.phi_dot_grid, [0.0])
  phi_and_phi_dot_points= np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  n = len(controllers)
  fig1, ax1s = plt.subplots(1,n)

  difference_array = np.zeros((GridPoints.len_phi_half_grid, GridPoints.len_phi_dot_grid))

  for idx in range(n):
    print("calculating basis of attraction for controller " + names[idx])
    t3 = datetime.now()
    controller = controllers[idx]

    success_array = np.zeros((GridPoints.len_phi_half_grid,
      GridPoints.len_phi_dot_grid))


    #only plot half of state space because it is symettric
    #this speeds it up

    for (i, phi) in enumerate(GridPoints.phi_half_grid):

      print("phi: " + str(phi))
      for (j, phi_dot) in enumerate(GridPoints.phi_dot_grid):

        (success, _ ) = runBicycleTest(stateflag = None, controller = controller,
        name = "", reward_flag = 14, simulation_duration= 2.0,
        isGraphing  = False, figObject = None, isPrinting = False,
        integrator_method = "fixed_step_RK4",
        USE_LINEAR_EOM = False, timestep = 1/50, starting_state3 = [phi, phi_dot, 0.0],
        v = v)

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
    ax1.set_yticks(np.arange(GridPoints.len_phi_half_grid))
    ax1.set_xticks(np.arange(GridPoints.len_phi_dot_grid))
    ax1.set_yticklabels(np.around(GridPoints.phi_half_grid, decimals=2))
    ax1.set_xticklabels(np.around(GridPoints.phi_dot_grid, decimals = 2))

    fig1.colorbar(im1)

    # np.savetxt("BasinOfAttraction/"+ names[idx] + "_BasisOfAttraction.csv", success_array,
    #   delimiter = ",")

    t4 = datetime.now()
    print(" calculated basis of attraction for controller " + names[idx] +
      " in " + str(t4-t3))

  fig2, ax2 = plt.subplots(1,1)
  #plot differences

  im2 = ax2.imshow(difference_array)
  ax2.set_title(" Difference between " + str(names[0])+ " and " + str(names[1]))
  ax2.set_ylabel("lean [rad]")
  ax2.set_xlabel("lean rate [rad/s]")
  ax2.set_yticks(np.arange(GridPoints.len_phi_half_grid))
  ax2.set_xticks(np.arange(GridPoints.len_phi_dot_grid))
  ax2.set_yticklabels(np.around(GridPoints.phi_half_grid, decimals=2))
  ax2.set_xticklabels(np.around(GridPoints.phi_dot_grid, decimals = 2))

  fig2.colorbar(im2)

  t2 = datetime.now()
  print("plotted basis of attraction in " + str(t2-t1) + " sec")

  plt.show()

# #name = "VI_r14_s6_a1"
# name = "VI_r14_a1_s16_v1_50episodes"
# VI_model = ValueIteration(state_grid_flag = 16, action_grid_flag = 1,
# reward_flag = 14, Ufile = "modelsB/"+name, use_only_continuous_actions = False,
# remake_table = False, step_table_integration_method = "fixed_step_RK4",
# USE_LINEAR_EOM = False, name = name, timestep = 1/50, v = 1.0)


# VI_model.init_controller(use_continuous_actions = True,
#   use_continuous_state_with_discrete_actions = True,
#   controller_integration_method = "fixed_step_RK4",
#   use_regression_model_of_table = False)

# plot_basin_of_attraction([LinearController.LinearController(getLQRGains("lqrd_3m_s")),
#   VI_model], ["linear r14 >>3m/s<<", "lqrd_1m_s"],  16, v = 1.0)

#use state grid points 16.2 so straight lines are straight
plot_basin_of_attraction([LinearController.LinearController(getLQRGains("lqrd_3m_s")),
   LinearController.LinearController(getLQRGains("lqrd_1m_s"))],
   ["linear_r14_3m/s", "linear_r14_1m/s"],  16.2, v = 3.0)

# plot_basin_of_attraction([LinearController.LinearController(getLQRGains("lqrd_2m_s"))],
#  ["linear r14"],  16)