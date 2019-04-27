import numpy as np
import rhs
import integrator
import matplotlib.pyplot as plt
import graph
from unpackState import *
from scipy.interpolate import interpn
import time
from pathlib import Path
from ControllerClass import *
from StateGridPoints import *
from numba import jit

class TableBased(Controller, StateGridPoints):

  # create points for actions based on action_grid_flag
  # store actions in action_grid
  def set_action_grid_points(self, action_grid_flag):

    if action_grid_flag == 0:
      self.action_grid = [-2, -1, 0, 1, 2]
    elif action_grid_flag == 1:
      self.action_grid = np.linspace(-2,2,11, endpoint=True)
    elif action_grid_flag == 2:
      self.action_grid = np.linspace(-2,2,101, endpoint=True)
    elif action_grid_flag == 3: #lol, improved my ability to count
      self.action_grid = np.linspace(-2,2,81, endpoint=True)
    elif action_grid_flag == 4:
      # Flag for continuous action space
      self.action_grid = None
    # 15 actions
    elif action_grid_flag == 5:
      self.action_grid = [-2, -1.5, -1, -.75, -.5, -.25, -.1, 0, .1, .25, .5, .75,  1, 1.5, 2]

    elif action_grid_flag == 6:
      self.action_grid = np.linspace(-5,5,26, endpoint=True)
    elif action_grid_flag == 7:
      self.action_grid = np.linspace(-5,5,51, endpoint=True)

    else:
      raise Exception("Invalid action_grid_flag: {}".format(action_grid_flag))

    if action_grid_flag == 4:
      self.num_actions = 0
    else:
      self.num_actions = len(self.action_grid)

  def __init__(self, state_grid_flag, action_grid_flag, reward_flag, USE_LINEAR_EOM,
    timestep = 1/50):
    StateGridPoints.set_state_grid_points(self, state_grid_flag)
    self.set_action_grid_points(action_grid_flag)
    self.reward_flag = reward_flag
    self.USE_LINEAR_EOM = USE_LINEAR_EOM
    self.timestep = timestep

    super(TableBased, self).__init__()

  #given: phi_index, phi_dot_index, delta_index
  #returns:continous, full - 8 variable, state for this index
  def state8_from_indicies(self, phi_index, phi_dot_index, delta_index):
    state3_index = (phi_index, phi_dot_index, delta_index)
    state3 = self.state_grid_points[state3_index]
    return state3_to_state8(state3)

  def state3_index_to_state3(self, state3_index):
    return self.state_grid_points[state3_index]

  #given: index of action in action_grid
  # return: action, (continous valued) steer rate command
  # def get_action_from_index(self, action_index):
  #   return self.action_grid[action_index]

  # return the 3-tuple of the indicies of the state grid point closest to state8
  #return a state3 variable
  def discretize(self, state8):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    closest_phi_point = np.abs(phi-self.phi_grid).argmin()
    closest_phi_dot_point = np.abs(phi_dot-self.phi_dot_grid).argmin()
    closest_delta_point = np.abs(delta-self.delta_grid).argmin()

    return (closest_phi_point, closest_phi_dot_point, closest_delta_point)

  #this function only works for states which are the state gridpoints.
  #this is useful for value Iteration
  #step table maps state indicies and action indicies to the next state

  def setup_step_table(self, reward_flag, remake_table,
    step_table_integration_method = "fixed_step_RK4"):

    if Path(self.step_table_file).is_file() and not remake_table:
      print("Loading step_table {} from file".format(self.Ufile))

      saved_step_table = np.genfromtxt(self.step_table_file, delimiter = ",")
      self.step_table = saved_step_table.reshape(self.len_phi_grid,
        self.len_phi_dot_grid, self.len_delta_grid, self.num_actions,3)


      saved_reward_table = np.genfromtxt(self.reward_file, delimiter = ",")
      self.reward_table = saved_reward_table.reshape(self.len_phi_grid,
        self.len_phi_dot_grid, self.len_delta_grid, self.num_actions)



    else:
      print("Making new step table {}".format(self.Ufile))
      t0 = time.time()
      self.step_table = np.zeros((self.len_phi_grid, self.len_phi_dot_grid,
        self.len_delta_grid, self.num_actions, 3))
      self.reward_table = np.zeros((self.len_phi_grid, self.len_phi_dot_grid,
        self.len_delta_grid, self.num_actions))
        #8 is number of variables in a continuous state

      for i_phi in range(self.len_phi_grid):
        for i_phi_dot in range(self.len_phi_dot_grid):
          for i_delta in range(self.len_delta_grid):

            state8 = self.state8_from_indicies(i_phi, i_phi_dot, i_delta)
              #don't use reward from stepping because that is reward for next state
              # s', and not current state, s (I think)

            for i_action in range(self.num_actions):

              action = self.action_grid[i_action]

              reward = get_reward(state8, action, reward_flag)
              self.reward_table[i_phi, i_phi_dot, i_delta, i_action] = reward

              #reward for step is reward for the current action and NEXT state
              new_state8, _, _ = step(state8, action, reward_flag,
                method = step_table_integration_method,
                USE_LINEAR_EOM = self.USE_LINEAR_EOM, timestep = self.timestep)
              new_state3 = state8_to_state3(new_state8)

              self.step_table[i_phi, i_phi_dot, i_delta, i_action] = new_state3

      np.savetxt(self.step_table_file,
        self.step_table.reshape(self.num_states*self.num_actions*3), delimiter = ",")
      np.savetxt(self.reward_file,
        self.reward_table.reshape(self.num_states*self.num_actions), delimiter = ",")
      t1 = time.time()
      print("Setup Step and Reward Tables in " + str(t1 - t0) + "sec")

  #given: a 3-tuple state3_index of the indicies for phi, phi_dot, delta
  #       the index of the action to take
  # return: state (continous, 8 varible) corresponding to taking the action
  # at action index in the state represented by state3_index
  def step_fast(self, state3_index, action_index):
    phi_i = state3_index[0]
    phi_dot_i = state3_index[1]
    delta_i = state3_index[2]

    return self.step_table[phi_i, phi_dot_i, delta_i, action_index]


#given: state (a state in the continous space)
#       u (an action in the continoue state)
# tstep_multiplier is the number of integration_timesteps for every one
# controller timestep. Should be an integer >= 1
#return: (state, reward, isDone)
@jit()
def step(state8, u, reward_flag, tstep_multiplier = 1,
  method = "fixed_step_RK4", USE_LINEAR_EOM = False, timestep = 1/50):

  new_state8 = integrator.integrate(state8, u, timestep, method = method,
    USE_LINEAR_EOM = USE_LINEAR_EOM)

  [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(new_state8)

  # check if bike has fallen, only need to do once a timestep
  if (np.abs(phi) >= np.pi/4):
    #print("Bike has fallen; Test Failure")
    isDone = True
  else:
    isDone = False

  reward = get_reward(state8, u, reward_flag)
  #reward = self.get_reward(new_state8, u, reward_flag)

  return (new_state8, reward, isDone)

#given: state8 - the state to get the reward of
#       reward_flag - dictates what reward shaping to use
# returns: reward - a nunber greater than or equal to  0
# reward = 0 iff the bike has fallen (phi > pi/4)
def get_reward(state8, action, reward_flag = 3):
  [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

  #REWARD_FOR_FALLING = 0
  #Do not change (if the bike falls, the utility
  # of that state is set to 0, ie ,no need to look at utilities going forward
  # so, the reward_for_falling should always be 0 and all other rewards should
  # be greater than 0, so when the bicycle does not fall, it gets a utility
  # greater than that if it falls).

  # test ifbike has fallen
  if (abs(phi) > np.pi/4):
    reward = 0
    #If reward is changed from 0, also change fill_value in the interpolator,
    #I should set them both to be dependent on 1 parameter
  else:
    if reward_flag == 0:
      reward =  1 #no shapping
    elif reward_flag == 1:
      reward = (1-(abs(phi))/2 - np.sign(phi)*phi_dot/20) #basic reward shaping
    elif reward_flag == 2:
      reward = 1/(phi**2+0.01) #add a little bit, so that we don't divide by 0
    #garantees reward for not falling down is greater than that for falling
    elif reward_flag == 3:
      reward = 5 - phi**2
    elif reward_flag == 4:
      reward = 3 - np.abs(phi)
    elif reward_flag == 5:
      reward = 5 - (np.abs(phi) + np.abs(phi_dot/4) + np.abs(delta))
    #incorperate actions
    elif reward_flag == 6:
       reward = 5 - phi**2 - 0.01*action**2
    # state flag 7 is good
    elif reward_flag == 7:
       reward = 5 - phi**2 - 0.001*action**2
    elif reward_flag == 8:
       reward = 5 - phi**2 - 0.0001*action**2
    elif reward_flag == 9:
       reward = 5 - phi**2 - .0005*action**2
    elif reward_flag == 10:
       reward = 5 - phi**2 - .002*action**2
    elif reward_flag == 11:
       reward = 5 - (phi**2 + phi_dot**2*1e-6 + delta**2*1e-6 + .001*action**2)
    #make reward flag constant larger to avoid assertion errors, hopefully
    elif reward_flag == 11.1:
       reward = 15 - (phi**2 + phi_dot**2*1e-6 + delta**2*1e-6 + .001*action**2)

     #use reward from well behaved LQR controller
    elif reward_flag == 12:
      reward = 10 - (phi**2 + 0.1*phi_dot**2* + 0.1*delta**2 + .001*action**2)
    elif reward_flag == 13:
      reward = 10 - (phi**2 + 0.1*phi_dot**2 + .001*action**2)
    elif reward_flag == 14:
      reward = 30 - (phi**2 + 0.05*phi_dot**2 + 0.05*delta**2 + 0.003*action**2)

    else:
      raise Exception("Invalid reward_flag: {}".format(reward_flag))

    if reward <= 0:
      print("reward: " + str(reward) + ", action: " + str(action))
    assert (reward > 0)

  return reward
