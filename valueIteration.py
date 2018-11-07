import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *
from scipy.interpolate import RegularGridInterpolator

class ValueIteration(TableBased):

  def __init__(self):
    super(ValueIteration, self).__init__()
    self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid))

  def get_reward(self,state, is_shaping = True):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    # test ifbike has fallen
    if (abs(phi) > np.pi/4):
      return 0
    else:
      if is_shaping:
        return (1-(abs(phi)) - np.sign(phi)*phi_dot/10) #basic reward shaping
      else:
        return 1 #no shapping

  # take in a (continuous) state
  def calc_best_action_and_utility(self, state, do_interpolation):

    Qtemp = np.zeros(self.num_actions)

    for action_index in range(self.num_actions):
      action = self.get_action_from_index(action_index)
      (new_state, _, _) = self.step(state, action)

      new_state3 = state_to_state3(new_state)
      Qtemp[action_index] = self.get_value(new_state3, do_interpolation)

   # print(Qtemp)

    max_Qtemp = np.max(Qtemp)
    best_action_index = np.argmax(Qtemp)

    return (best_action_index, max_Qtemp)

  # called by simulate state.
  # given state_grid_point_index (a discritized state 3 tuple).
  # return the index of the action to take
  def act_index(self, state_grid_point_index, epsilon):

    state3 = self.state_grid_points[state_grid_point_index]
    state = state3_to_state(state3)

    (best_action_index, _) = self.calc_best_action_and_utility(state, False)
    return best_action_index

  # return a value for the (continous) new_state3
  #interpolater
  def get_value(self,new_state3, do_interpolation):
    #interpolate
    #switching the interpolator makes the code take much longer.
    #I should only set up the interpolater once per episode
    if do_interpolation:
      return self.itp([new_state3[0],new_state3[1],new_state3[2]])

    else:
    #dont interpolate
      new_state3_index = self.discretize(state3_to_state(new_state3))
      return self.U[new_state3_index]


  def train(self, gamma = 0.95, num_episodes = 30, state_flag = 0,
    do_interpolation = True):

    n_episode = 0

    while (n_episode < num_episodes):

      if do_interpolation:
        self.itp = RegularGridInterpolator(\
          (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
          bounds_error = False, fill_value = 0)
          # false bounds error causes us to extrapolate values out of range

      #exhaustively loop through all states
      for phi_index in range(self.len_phi_grid):
        for phi_dot_index in range(self.len_phi_dot_grid):
          for delta_index in range (self.len_delta_grid):

            state3_index = (phi_index, phi_dot_index, delta_index)
            state3 = self.state_grid_points[state3_index]
            state = state3_to_state(state3)

            (_, max_Qtemp) = \
             self.calc_best_action_and_utility(state,do_interpolation)


            #If the bike fell down, set the value to 0
            if (self.get_reward(state) == 0):
              self.U[state3_index] = 0
            else:
              self.U[state3_index] = self.get_reward(state) + gamma*max_Qtemp

      n_episode += 1
      print('Epsiode: ' + str(n_episode))

    print("done trianing")
    print(self.U)
    np.savetxt("valueIteration_U.csv", self.U.reshape(self.num_states), \
        delimiter = ",")

  def test(self, Ufile = "valueIteration_U.csv", tmax = 10, state_flag = 0,
      gamma = 1):

    epsilon = 0; alpha = 0

    saved_U = np.genfromtxt(Ufile, delimiter = ",")
    self.U = saved_U.reshape(self.len_phi_grid, self.len_phi_dot_grid, \
        self.len_delta_grid)

    reward, time, states, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, True, state_flag)

    print("testing reward: " + str(reward) + ", testing time: " + str(time))
    graph.graph(states, motorCommands)


VIteration_model = ValueIteration()
VIteration_model.train()

VIteration_model.test(Ufile = "valueIteration_U.csv", tmax = 10, state_flag = 0,
      gamma = 1)