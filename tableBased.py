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

  #3 tuple. one dimension for each state variable
  state_grid_points = (phi_grid, phi_dot_grid, delta_grid)

  timestep = 1/50

  def __init__(self):
    pass

  def action_from_index(self, action_index):
    return self.action_grid[action_index]

  # return the 3-tuple of the indicies of the state grid point closest to state
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