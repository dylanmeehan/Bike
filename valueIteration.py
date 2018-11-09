import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *
from scipy.interpolate import RegularGridInterpolator

class ValueIteration(TableBased):

  def __init__(self, state_grid_flag, action_grid_flag):
    super(ValueIteration, self).__init__(state_grid_flag, action_grid_flag)
    self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid))
    self.setup_step_table()

    #TODO: precompute reward function. Set up table (similiar to step_table), so
    #       that we don't compute the reward for a state every time. We only
    #       look up the reward

  #given: state3_index: the index of a point in the descritized table
  #     do_intepolations: boolean to decide to interpolate or not
  #return: (best_action_index, best_action_utility) for that state.
  #`best_action_index is the index of the action which has the highest utility
  # best_action_utility is the utility of that action
  def calc_best_action_and_utility(self, state3_index, do_interpolation):

    Qtemp = np.zeros(self.num_actions)

    state3 = self.state_grid_points[state3_index]

    for action_index in range(self.num_actions):

      new_state8 = self.step_fast(state3_index, action_index)
      new_state3 = state8_to_state3(new_state8)

      Qtemp[action_index] = self.get_value(new_state3, do_interpolation)

    best_action_utility = np.max(Qtemp)
    best_action_index = np.argmax(Qtemp)

    return (best_action_index, best_action_utility)

  # given: state3_index (a discritized state 3 tuple).
  # return: the index of the best action to take
  def act_index(self, state3_index, epsilon):

    (best_action_index, _) = self.calc_best_action_and_utility(state3_index, False)
    return best_action_index

  # return a value for the (continous) new_state3
  # do_interpolation is a boolean to decide if to interpolate valuesS
  def get_value(self,new_state3, do_interpolation):

    #the interpolator (itp) is initialized in train, one for each episode.
    # this saves on the overhead of creating an interpolator
    if do_interpolation:
      return self.itp([new_state3[0],new_state3[1],new_state3[2]])

    else:
    #dont interpolate
      new_state3_index = self.discretize(state3_to_state8(new_state3))
      return self.U[new_state3_index]

  #trains a valueIteration, table-based mode.
  #when training finishes, utilities are stored in a csv
  def train(self, gamma = 0.95, num_episodes = 30, state_flag = 0,
    do_interpolation = True):

    n_episode = 0

    while (n_episode < num_episodes):

      if do_interpolation:
        self.itp = RegularGridInterpolator(\
          (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
          bounds_error = False, fill_value = 0)
          # false bounds error causes us to extrapolate values out of range
          #fill_value = 0, sets the value outside of the interpolation range to 0
          # thus, if the bicycle gets no reward for a state outside of the grid
          # this ensures bad states have a reward of 0 (as desired)

      #exhaustively loop through all states
      for phi_i in range(self.len_phi_grid):
        for phi_dot_i in range(self.len_phi_dot_grid):
          for delta_i in range (self.len_delta_grid):

            state3_index = (phi_i, phi_dot_i, delta_i)
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)

            (_, best_utility) = \
             self.calc_best_action_and_utility(state3_index,do_interpolation)

            #If the bike fell down, set the value to 0
            reward = self.get_reward(state8)
            if (reward == 0):
              #print("entered top of if statement")
              self.U[state3_index] = 0
            else:
              self.U[state3_index] = reward + gamma*best_utility

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

    reward, time, states8, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, True, state_flag)

    print("testing reward: " + str(reward) + ", testing time: " + str(time))
    graph.graph(states8, motorCommands)


VIteration_model = ValueIteration(state_grid_flag = 0, action_grid_flag = 0)
VIteration_model.train()

VIteration_model.test(Ufile = "valueIteration_U.csv", tmax = 10, state_flag = 1,
      gamma = 3)