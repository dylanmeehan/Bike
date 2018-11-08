import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from scipy.interpolate import interpn

class TableBased(object):

  timestep = 1/50

  def set_action_grid_points(self, action_grid_flag):

    if action_grid_flag == 0:
      self.action_grid = [-2, -1, 0, 1, 2]

    else:
      raise Exception("Invalid state_grid_flag: {}".format(state_grid_flag))

    self.num_actions = len(self.action_grid)

  #discritize each of the state variables. Construct self.state_grid_points
  # which is a meshgrid of these points
  def set_state_grid_points(self, state_grid_flag):

    #generate grid in which to discritize states
    if state_grid_flag == 0:
      self.phi_grid = [-.8, -.6, -.4, -.28, -.22, -.16, -.1, -.06, -.02, 0, \
        .02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      self.phi_dot_grid = [-1, -.7, -.4,  -.3, -.2, -.1, -.05, -.02, 0,\
        .02, .05, .1, .2, .3, .4,  .7, 1]
      self.delta_grid = [-1, -.7, -.4,  -.3, -.2, -.1, -.05, -.02, 0,\
        .02, .05, .1, .2, .3, .4,  .7, 1]

    else:
      raise Exception("Invalid state_grid_flag: {}".format(state_grid_flag))

    # calculate lengths once and store their values
    self.len_phi_grid = len(self.phi_grid)
    self.len_phi_dot_grid = len(self.phi_dot_grid)
    self.len_delta_grid = len(self.delta_grid)
    self.num_states = self.len_phi_grid*self.len_phi_dot_grid*self.len_delta_grid


    # generate a 3D grid of the points in our table
    # a mesh where each point is a 3 tuple. one dimension for each state variable
    phi_points, phi_dot_points,  delta_points  = \
      np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing='ij')
    self.state_grid_points = np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  def __init__(self, state_grid_flag, action_grid_flag):
    self.set_state_grid_points(state_grid_flag)
    self.set_action_grid_points(action_grid_flag)

  #given: phi_index, phi_dot_index, delta_index
  #returns:continous, full - 8 variable, state for this index
  def state8_from_indicies(self, phi_index, phi_dot_index, delta_index):
    state3_index = (phi_index, phi_dot_index, delta_index)
    state3 = self.state_grid_points[state3_index]
    return state3_to_state8(state3)

  # return action a (continous valued) steer rate command
  def get_action_from_index(self, action_index):
    return self.action_grid[action_index]

  # return the 3-tuple of the indicies of the state grid point closest to state
  #return a state3 variable
  def discretize(self, state8):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    #we don't know when Q has been updated, so make a new interpolator object

    closest_phi_point = np.abs(phi-self.phi_grid).argmin()
    closest_phi_dot_point = np.abs(phi_dot-self.phi_dot_grid).argmin()
    closest_delta_point = np.abs(delta-self.delta_grid).argmin()

    return (closest_phi_point, closest_phi_dot_point, closest_delta_point)

  #getStartingState returns an 8 value continuous state.
  #this is not affected by the discritization grid used for the table methods
  def getStartingState8(self, state_flag = 0):
    starting_states = {
      0: np.array([0, 0, 0, 0.01, 0, 0, 0, 3]),
      1: np.array([0, 0, 0, np.pi/32, 0, 0, 0, 3]),
      2: np.array([0, 0, 0, np.random.uniform(-np.pi/16, np.pi/16) , 0, 0, 0, 3]),
      3: np.array([0, 0, 0, np.pi/16, 0, 0, 0, 3]),
    }
    return starting_states[state_flag]

  #this function only works for states which are the state gridpoints.
  #this is mostly useful for value Iteration
  def setup_step_table(self):
    self.step_table = np.zeros((self.len_phi_grid, self.len_phi_dot_grid,
      self.len_delta_grid, self.num_actions, 8))
      #8 is number of variables in a continuous state

    for i_phi in range(self.len_phi_grid):
      for i_phi_dot in range(self.len_phi_dot_grid):
        for i_delta in range(self.len_delta_grid):
          for i_action in range(self.num_actions):
            state8 = self.state8_from_indicies(i_phi, i_phi_dot, i_delta)
            action = self.action_grid[i_action]

            new_state8, reward, _ = self.step(state8, action)
            self.step_table[i_phi, i_phi_dot, i_delta, i_action] = new_state8

  #given: a 3-tuple state3_index of the indicies for phi, phi_dot, delta
  #       the index of the action to take
  # return: state (continous, 8 varible)
  def step_fast(self, state3_index, action_index):
    phi_i = state3_index[0]
    phi_dot_i = state3_index[1]
    delta_i = state3_index[2]

    return self.step_table[phi_i, phi_dot_i, delta_i, action_index]

  #given: state (a state in the continous space)
  #       u (an action in the continoue state)
  #return: (state, reward, isDone)
  def step(self, state8, u):
    # take in a state and an action
    zdot = rhs.rhs(state8,u)

    #update state. Euler Integration
    prevState8 = state8
    state8 = state8 + zdot*self.timestep

    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    if (np.abs(phi) >= np.pi/4):
      #print("Bike has fallen; Test Failure")
      isDone = True
      reward = 0
    else:
      reward = self.timestep
      isDone = False

    return (state8, reward, isDone)

  # state_flag = 0 is a flag. Tells function to call getStartingState()
  # to give a proper state, pass in a state vector
  def simulate_episode(self, epsilon, gamma, alpha, tmax,
    isTesting, state_flag = 0):

    state8 = self.getStartingState8(state_flag)

    done = False
    total_reward = 0
    total_time = 0

    # return index of closest point to phi, phi_dot, and delta
    state_grid_point_index = self.discretize(state8)

    maxNumTimeSteps = int(tmax/self.timestep)+1

    if isTesting:
      #create arrays before loop
      success = True
      numStates8 = state8.size
      states8 = np.zeros([maxNumTimeSteps, numStates8])
      motorCommands = np.zeros([maxNumTimeSteps, 1])

      #initialize starting values of arrays
      states8[1,:] = state8
      motorCommands[1] = 0

    count = 0;
    while( (count < maxNumTimeSteps) and (not done)):
      action_index = self.act_index(state_grid_point_index, epsilon)
      action = self.get_action_from_index(action_index)

      new_state8, reward, done = self.step(state8, action)
      new_state_grid_point_index = self.discretize(new_state8)

      if (not isTesting):
        self.update_Q(state_grid_point_index, reward, action_index, \
          new_state_grid_point_index, done, alpha, gamma)

      total_reward += reward
      if (not done):
        total_time += self.timestep

      state8 = new_state8
      state_grid_point_index = new_state_grid_point_index

      if isTesting:
        states8[count,:] = state8
        motorCommands[count] = action

      count += 1

    if isTesting:
    #trim off zero values. This avoids the graph drawing a line to the origin.
      states8 = states8[:count,:]
      motorCommands = motorCommands[:count]

    if (not isTesting):
      states8 = False; motorCommands = False;

    return [total_reward, total_time, states8, motorCommands]
