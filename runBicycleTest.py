import numpy as np
import matplotlib.pyplot as plt
import rhs
import LinearController
import graph
from unpackState import *
from tableBased import *
import integrator
import scipy.integrate as inter
import time

# use Euler Integration to simulate a bicycle
def runBicycleTest(stateflag, controller, name, reward_flag, simulation_duration,
  isGraphing  = True, figObject = None, tstep_multiplier = 1,
  integrator_method = "fixed_step_RK4",
  USE_LINEAR_EOM = False, timestep = 1/50, starting_state3 = None, isPrinting= True):

  if stateflag == None:
    state8 = np.array(state3_to_state8(starting_state3))
  else:
    state8 = getStartingState8(stateflag)

  #controllers get a reward every timestep, so normalize the rewards by a canonical
  # timestep so that we can compare rewards between controllers with different
  # timesteps
  canonical_timestep = 1/50

  if not controller.is_initialized:
    print("# # # " + name + " is not intialized # # # \n skipping " + name)
    return figObject

  t_test1 = time.time()



  numTimeSteps = int(simulation_duration/timestep)+1

  #create arrays before loop
  success = True
  numStates = state8.size
  states = np.zeros([numTimeSteps, numStates])
  motorCommands = np.zeros([numTimeSteps, 1])

  #initialize starting values of arrays
  # states8[1,:] = state8
  # motorCommands[1] = 0

  count = 0
  cum_reward = 0
  sim_time = 0

  is_fallen = False

  while( (count < numTimeSteps) and not (is_fallen)):

    #calculate control action
    u = controller.act(state8, timestep)

    new_state8, reward, is_fallen = step(state8, u, reward_flag, tstep_multiplier = 1,
      method = "fixed_step_RK4", USE_LINEAR_EOM = USE_LINEAR_EOM, timestep = timestep)

    states[count,:] = state8
    motorCommands[count] = u

    if not is_fallen:
      sim_time += timestep
    cum_reward += reward
    count += 1



    state8 = new_state8

  states = states[:count,:]
  motorCommands = motorCommands[:count]

  cum_reward = cum_reward / (canonical_timestep/timestep)
  success = not is_fallen

  #### copied from simulate_episode. used to calculate which states are inside
  # the last gridpoint of a VI model
  # points_inside_last_gridpoint = []
  # if plot_is_inside_last_gridpoint:
  #   points_inside_last_gridpoint = \
  #     self.calculate_points_inside_last_gridpoint(states8)


  if isPrinting:
    print(name + " success: " + str(success) + ", cumulative reward:" + str(cum_reward) + ",  time in simulation: "
          + str(sim_time))


    t_test2 = time.time()
    print("Tested " + name + " in " + str(t_test2-t_test1) + " sec of computer time")

  if isGraphing:
    figObject = graph.graph(states, motorCommands, figObject, [], name)


  return success, figObject