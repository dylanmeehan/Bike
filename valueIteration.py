import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *

class ValueIteration(TableBased):

  def __init__(self):
    super(ValueIteration, self).__init__()
    self.U = np.zeros(self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid)


  def get_reward(self,state):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    # test ifbike has fallen
    if (abs(phi) > np.pi/4):
      return 0
    else:
      return 1


  # return a value for the (continous) new state
  def get_value(self,new_state):
    new_state_index = self.discritize(new_state)
    return self.U[new_state_index]


  def train(self, gamma, num_episodes):

    n_epsiode = 0

    while (n_episode < num_epsiodes):
      #exhaustively loop through all states
      for phi_index in range(self.len_phi_grid):
        for phi_dot_index in range(self.len_phi_dot_grid):
          for delta_index in range (self.len_delta_grid):

            Qtemp = np.zeros(self.num_actions)
            state_index = (phi_index, phi_dot_index, delta_index)
            state = state_grid_points[state_index]

            for action_index in range(self.num_actions):
              action = self.action
              (new_state, _, _) = self.step(state, action)

              Qtemp[action_index] = gamma*get_value(new_state)

            self.U[state_index] = self.get_reward(state) + np.max(Qtemp)

      n_episode += 1
      print('Epsiode: ' + str(n_episode))

      np.savetext("valueIteration_A.csv", self.U.reshape(self.num_states), \
        delimiter = ",")


