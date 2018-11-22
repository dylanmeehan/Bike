import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


class ValueIteration(TableBased):

  def __init__(self, state_grid_flag, action_grid_flag, reward_flag,
    Ufile = "valueIteration_U.csv"):

    super(ValueIteration, self).__init__(state_grid_flag, action_grid_flag,
      reward_flag)

    self.setup_step_table(reward_flag)

    self.Ufile = Ufile

    if Path(self.Ufile).is_file():
      saved_U = np.genfromtxt(Ufile, delimiter = ",")
      self.U = saved_U.reshape(self.len_phi_grid, self.len_phi_dot_grid,
          self.len_delta_grid)
    else:
      self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid,
        self.len_delta_grid))


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

    self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid))

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
    #print(self.U)
    np.savetxt(self.Ufile, self.U.reshape(self.num_states), delimiter = ",")

  def test(self, tmax = 10, state_flag = 0,
      gamma = 1, figObject = None):

    epsilon = 0; alpha = 0

    reward, time, states8, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, True, state_flag)

    print("VALUE ITERATION: testing reward: " + str(reward) + ", testing time: "
      + str(time))

    figObject = graph.graph(states8, motorCommands, figObject)
    return figObject

  def heatmap_value_function(self):
    #print("phi num = " + str(self.len_phi_grid) + ", phi dot num = "  +
    #  str(self.len_phi_dot_grid) + ", delta num: " + str(self.len_delta_grid))
    #print("U size = " + str(self.U.shape))

    # figure 1
    fig1, ax1 = plt.subplots(1,1)

    phi_vs_phidot = np.mean(self.U, axis = 2)
    #print("phi vs phidot shape: " + str(phi_vs_phidot.shape))

    im1 = ax1.imshow(phi_vs_phidot)
    ax1.set_title("Utility (averaged over delta)")
    ax1.set_ylabel("phi [rad]")
    ax1.set_xlabel("phi_dot [rad/s]")
    ax1.set_yticks(np.arange(self.len_phi_grid))
    ax1.set_xticks(np.arange(self.len_phi_dot_grid))
    ax1.set_yticklabels(self.phi_grid)
    ax1.set_xticklabels(self.phi_dot_grid)

    fig1.colorbar(im1)

    #figure 2
    fig2, ax2 = plt.subplots(1,1)
    phi_vs_delta = np.mean(self.U, axis = 1)
    #print("phi vs phidot shape: " + str(phi_vs_delta))

    im2 = ax2.imshow(phi_vs_delta)
    ax2.set_title("Utility (averaged over phidot)")
    ax2.set_ylabel("phi [rad]")
    ax2.set_xlabel("delta [rad]")
    ax2.set_yticks(np.arange(self.len_phi_grid))
    ax2.set_xticks(np.arange(self.len_delta_grid))
    ax2.set_yticklabels(self.phi_grid)
    ax2.set_xticklabels(self.delta_grid)

    fig2.colorbar(im2)

    #figure 3
    fig3, ax3 = plt.subplots(1,1)
    phidot_vs_delta = np.mean(self.U, axis = 0)
    #print("phi vs phidot shape: " + str(phidot_vs_delta))

    im3 = ax3.imshow(phidot_vs_delta)
    ax3.set_title("Utility (averaged over phi)")
    ax3.set_ylabel("phi_dot [rad/s]")
    ax3.set_xlabel("delta [rad]")
    ax3.set_yticks(np.arange(self.len_phi_dot_grid))
    ax3.set_xticks(np.arange(self.len_delta_grid))
    ax3.set_yticklabels(self.phi_dot_grid)
    ax3.set_xticklabels(self.delta_grid)

    fig3.colorbar(im3)

    #
    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

  def heatmap_of_policy(self, option):

    policy = np.zeros(self.U.shape)

    for phi_i in range(self.len_phi_grid):
        for phi_dot_i in range(self.len_phi_dot_grid):
          for delta_i in range (self.len_delta_grid):

            state3_index = (phi_i, phi_dot_i, delta_i)

            action_index = self.act_index(state3_index, epsilon = 0)
            #self.act_index returns which action to take. defined for each model.
            action = self.get_action_from_index(action_index)

            policy[phi_i, phi_dot_i, delta_i] = action

    get_middle = lambda array: array[int((len(array)-1)/2)]

    fig1, ax1 = plt.subplots(1,1)

    if option == "average":
      phi_vs_phidot = np.mean(policy, axis = 2)
    elif option == "zero":
      phi_vs_phidot = np.apply_along_axis(get_middle, axis=2, arr= policy)
    #print("phi vs phidot shape: " + str(phi_vs_phidot.shape))

    im1 = ax1.imshow(phi_vs_phidot)

    if option == "average":
      ax1.set_title("Policy (averaged over delta)")
    elif option == "zero":
      ax1.set_title("Policy (with delta=0)")
    ax1.set_ylabel("phi [rad]")
    ax1.set_xlabel("phi_dot [rad/s]")
    ax1.set_yticks(np.arange(self.len_phi_grid))
    ax1.set_xticks(np.arange(self.len_phi_dot_grid))
    ax1.set_yticklabels(self.phi_grid)
    ax1.set_xticklabels(self.phi_dot_grid)

    fig1.colorbar(im1)

    #figure 2
    fig2, ax2 = plt.subplots(1,1)

    if option == "average":
      phi_vs_delta = np.mean(policy, axis = 1)
    elif option == "zero":
      phi_vs_delta = np.apply_along_axis(get_middle, axis=1, arr= policy)
    #print("phi vs phidot shape: " + str(phi_vs_delta))

    im2 = ax2.imshow(phi_vs_delta)
    if option == "avearge":
      ax2.set_title("Policy (averaged over phidot)")
    elif option == "zero":
      ax2.set_title("Policy (with phi_dot=0)")
    ax2.set_ylabel("phi [rad]")
    ax2.set_xlabel("delta [rad]")
    ax2.set_yticks(np.arange(self.len_phi_grid))
    ax2.set_xticks(np.arange(self.len_delta_grid))
    ax2.set_yticklabels(self.phi_grid)
    ax2.set_xticklabels(self.delta_grid)

    fig2.colorbar(im2)

    #figure 3
    fig3, ax3 = plt.subplots(1,1)

    if option == "average":
      phidot_vs_delta = np.mean(policy, axis = 0)
    elif option == "zero":
      phidot_vs_delta = np.apply_along_axis(get_middle, axis=0, arr= policy)
    #print("phi vs phidot shape: " + str(phidot_vs_delta))

    im3 = ax3.imshow(phidot_vs_delta)
    if option == "average":
      ax3.set_title("Policy (averaged over phi)")
    elif option == "zero":
      ax3.set_title("Policy (with phi=0)")
    ax3.set_ylabel("phi_dot [rad/s]")
    ax3.set_xlabel("delta [rad]")
    ax3.set_yticks(np.arange(self.len_phi_dot_grid))
    ax3.set_xticks(np.arange(self.len_delta_grid))
    ax3.set_yticklabels(self.phi_dot_grid)
    ax3.set_xticklabels(self.delta_grid)

    fig3.colorbar(im3)

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


# VIteration_model = ValueIteration(state_grid_flag = 0, action_grid_flag = 0)
# VIteration_model.train()

# VIteration_model.test(Ufile = "valueIteration_U.csv", tmax = 10, state_flag = 1,
#       gamma = 3)