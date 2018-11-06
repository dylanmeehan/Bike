import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from scipy.interpolate import interpn

class TableBased(object):

  #descretize action space
  action_grid = [-2, -1, 0, 1, 2]
  num_actions = len(action_grid)

  #descritized states
  phi_grid = [-.8, -.6, -.4, -.28, -.22, -.16, -.1, -.06, -.02, 0, \
    .02, .06, .1, .16, .22, .28, .4, .6, .8 ]
  len_phi_grid = len(phi_grid)
  phi_dot_grid = [-1, -.7, -.4,  -.3, -.2, -.1, -.05, -.02, 0,\
    .02, .05, .1, .2, .3, .4,  .7, 1]
  len_phi_dot_grid = len(phi_dot_grid)
  delta_grid = [-1, -.7, -.4,  -.3, -.2, -.1, -.05, -.02, 0,\
    .02, .05, .1, .2, .3, .4,  .7, 1]
  len_delta_grid = len(delta_grid)
  num_states = len_phi_grid * len_phi_dot_grid * len_delta_grid

  # make stategrid.
  # a mesh where each point is a 3 tuple. one dimension for each state variable
  phi_points, phi_dot_points,  delta_points  = \
    np.meshgrid(phi_grid, phi_dot_grid, delta_grid, indexing='ij')
  state_grid_points = np.rec.fromarrays([phi_points, phi_dot_points, delta_points],\
    names='phi_points,phi_dot_points,delta_points')

  timestep = 1/50

  def __init__(self):
    pass

  def get_action_from_index(self, action_index):
    return self.action_grid[action_index]

  # return the 3-tuple of the indicies of the state grid point closest to state
  #return a state3 variable
  def discretize(self, state):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    #we don't know when Q has been updated, so make a new interpolator object

    closest_phi_point = np.abs(phi-self.phi_grid).argmin()
    closest_phi_dot_point = np.abs(phi_dot-self.phi_dot_grid).argmin()
    closest_delta_point = np.abs(delta-self.delta_grid).argmin()

    return (closest_phi_point, closest_phi_dot_point, closest_delta_point)

  def getStartingState(self, state_flag = 0):
    starting_states = {
      0: np.array([0, 0, 0, 0.01, 0, 0, 0, 3]),
      1: np.array([0, 0, 0, np.pi/32, 0, 0, 0, 3]),
      2: np.array([0, 0, 0, np.random.uniform(-np.pi/16, np.pi/16) , 0, 0, 0, 3])
    }
    return starting_states[state_flag]

  #given: state (a state in the continous space)
  #       u (an action in the continoue state)
  #return: (state, reward, isDone)
  def step(self, state, u):
    # take in a state and an action
      zdot = rhs.rhs(state,u)

      #update state. Euler Integration
      prevState = state
      state = state + zdot*self.timestep

      [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

      if (np.abs(phi) >= np.pi/4):
        #print("Bike has fallen; Test Failure")
        isDone = True
        reward = 0
      else:
        reward = self.timestep
        isDone = False

      return (state, reward, isDone)

  # state_flag = 0 is a flag. Tells function to call getStartingState()
  # to give a proper state, pass in a state vector
  def simulate_episode(self, epsilon, gamma, alpha, tmax,
    isTesting, state_flag = 0):

    state = self.getStartingState(state_flag)

    done = False
    total_reward = 0
    total_time = 0

    # return index of closest point to phi, phi_dot, and delta
    state_grid_point_index = self.discretize(state)

    maxNumTimeSteps = int(tmax/self.timestep)+1

    if isTesting:
      #create arrays before loop
      success = True
      numStates = state.size
      states = np.zeros([maxNumTimeSteps, numStates])
      motorCommands = np.zeros([maxNumTimeSteps, 1])

      #initialize starting values of arrays
      states[1,:] = state
      motorCommands[1] = 0

    count = 0;
    while( (count < maxNumTimeSteps) and (not done)):
      action_index = self.act_index(state_grid_point_index, epsilon)
      action = self.get_action_from_index(action_index)

      new_state, reward, done = self.step(state, action)
      new_state_grid_point_index = self.discretize(new_state)

      if (not isTesting):
        self.update_Q(state_grid_point_index, reward, action_index, \
          new_state_grid_point_index, done, alpha, gamma)

      total_reward += reward
      if (not done):
        total_time += self.timestep

      state = new_state
      state_grid_point_index = new_state_grid_point_index

      if isTesting:
        states[count,:] = state
        motorCommands[count] = action

      count += 1

    if isTesting:
    #trim off zero values. This avoids the graph drawing a line to the origin.
      states = states[:count,:]
      motorCommands = motorCommands[:count]

    if (not isTesting):
      states = False; motorCommands = False;

    return [total_reward, total_time, states, motorCommands]
