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


  def get_reward(self,state):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    # test ifbike has fallen
    if (abs(phi) > np.pi/4):
      return 0
    else:
      return 1

  # called by simulate state.
  # given state_grid_point_index (a discritized state 3 tuple).
  # return the index of the action to take
  def act_index(self, state_grid_point_index, epsilon):
    print("utilities: " + str(self.U[state_grid_point_index]))
    print("action: " + str(np.argmax(self.U[state_grid_point_index])))
    return np.argmax(self.U[state_grid_point_index])

  # return a value for the (continous) new state
  #interpolater
  def get_value(self,new_state3):
    #interpolate
    #switching the interpolator makes the code take much longer.
    #I should only set up the interpolater once per episode
    #return self.itp([new_state3[0],new_state3[1],new_state3[2]])

    #dont interpolate
    new_state3_index = self.discretize(state3_to_state(new_state3))
    return self.U[new_state3_index]


  def train(self, gamma = 0.99, num_episodes = 10, state_flag = 0):

    n_episode = 0

    while (n_episode < num_episodes):

      #self.itp = RegularGridInterpolator(\
      #  (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U)
      count = 0
      #exhaustively loop through all states
      for phi_index in range(self.len_phi_grid):
        for phi_dot_index in range(self.len_phi_dot_grid):
          for delta_index in range (self.len_delta_grid):

            Qtemp = np.zeros(self.num_actions)

            state3_index = (phi_index, phi_dot_index, delta_index)
            state3 = self.state_grid_points[state3_index]
            state = state3_to_state(state3)

            for action_index in range(self.num_actions):
              action = self.get_action_from_index(action_index)
              (new_state, _, _) = self.step(state, action)

              new_state3 = self.state_grid_points[state3_index]
              Qtemp[action_index] = gamma*self.get_value(new_state3)

            print(Qtemp)


            count += 1
            #If the bike fell down, set the reward to 0
            if (self.get_reward(state) == 0):
              self.U[state3_index] = 0
            else:
              self.U[state3_index] = self.get_reward(state) + np.max(Qtemp)



      n_episode += 1
      print('Epsiode: ' + str(n_episode) + "count: " + str(count))

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