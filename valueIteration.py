import numpy as np
import rhs
import matplotlib.pyplot as plt
import matplotlib as mpl
import graph
import parameters as params
from unpackState import *
from tableBased import *
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import scipy.optimize as opt
import LinearController

#set default font size
mpl.rcParams['font.size']=14

class ValueIteration(TableBased):

  def __init__(self, state_grid_flag, action_grid_flag, reward_flag,
    Ufile = "models/valueIteration_U.csv", use_only_continuous_actions = False):

    super(ValueIteration, self).__init__(state_grid_flag, action_grid_flag,
      reward_flag)

    if not use_only_continuous_actions:
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
  # best_action_index is the index of the action which has the highest utility
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

  #
  def continuous_utility_function(self, state8, u):

    (new_state8, reward, isDone) = self.step(state8, u, self.reward_flag)

    new_state3 = state8_to_state3(new_state8)
    utility = self.get_value(new_state3, do_interpolation = True)
    #does NOT calculate the reward at new_state3 (ie, does not calculate R(s'))

    return utility

  def calculate_points_inside_last_gridpoint(self, states8):
    ##### CALCULATE WHEN THE CONTROLLER IS INSIDE THE LAST GRID POINT #######
    #this code kinda duplicates the code inside graph. I should figure out
    #how to better structure this

    [ts, xs, ys, phis, psis, deltas, phi_dots, vs] =  \
      np.apply_along_axis(unpackState, 1, states8).T
    points_inside_last_gridpoint = np.zeros(len(phis))

    for t in range(len(ts)):
      #check if we are inside the last box
      if (abs(phis[t])<self.smallest_phi and abs(phi_dots[t])<self.smallest_phi_dot
        and abs(deltas[t])<self.smallest_delta):
        points_inside_last_gridpoint[t] = 1
      #else is already set to zero

    return points_inside_last_gridpoint

  #always do interpolation
  def calc_best_action_and_utility_continuous(self, state3):

    #state3 = self.state_grid_points[state3_index]
    state8 = state3_to_state8(state3)

    #we need to minimiza something. minimizning negations of the utility function
    # is equivalent to maximizing the utilty function
    negated_utility_fun = lambda u: -1*self.continuous_utility_function(state8, u)

    # find the action which maximizes the utility function

    OptimizeResult = opt.minimize(negated_utility_fun, x0=0,
      method = 'Powell', tol = 1e-4, options = {'xtol': 1e-4})
    #'Powell' method gave no failures
    #OptimizeResult = opt.minimize(negated_utility_fun, x0=0, method = 'TNC',
    # tol = 1e-3, options = {'xtol': 1e-3}, bounds = ((-3,3),))

    if not OptimizeResult.success:
      print("************* OPTIMIZER FAILED ***************")
      print(OptimizeResult)
    #print(OptimizeResult.x)
    u = OptimizeResult.x

    #clip u= delta dot= steer rate
    if u > params.MAX_STEER_RATE:
      u = params.MAX_STEER_RATE
    elif u < -params.MAX_STEER_RATE:
      u = -params.MAX_STEER_RATE

    best_action_utility = self.continuous_utility_function(state8, u)
    #print(u)

    return (u, best_action_utility)

  def get_action_continuous(self, state8, epsilon = 0):
    state3 = state8_to_state3(state8)
    (u, _) = self.calc_best_action_and_utility_continuous(state3)
    return u

  # given: state3_index (a discritized state 3 tuple).
  # return: the index of the best action to take
  def act_index(self, state3_index, epsilon):

    (best_action_index, _) = self.calc_best_action_and_utility(state3_index, False)
    return best_action_index

  # return a value for the (continuous) new_state3
  # do_interpolation is a boolean to decide if to interpolate valuesS
  def get_value(self,new_state3, do_interpolation= True):

    #the interpolator (itp) is initialized in train, one for each episode.
    # this saves on the overhead of creating an interpolator
    if do_interpolation:
      #commented out calculating reward for state we got to by interpolating
      #add reward_at_new_state3 to try to differentiate states
      #new_state8 = state3_to_state8(new_state3)
      #reward_at_new_state3 =  self.get_reward(new_state8, self.reward_flag)
      utility_from_new_state3 = self.itp([new_state3[0],new_state3[1],new_state3[2]])
      return utility_from_new_state3 #+ reward_at_new_state3

    else:
    #dont interpolate
      new_state3_index = self.discretize(state3_to_state8(new_state3))
      return self.U[new_state3_index]

  #trains a valueIteration, table-based mode.
  #when training finishes, utilities are stored in a csv
  # if continuous actions is true, do_interpolation must be true
  def train(self, gamma = 0.95, num_episodes = 30,
    do_interpolation = True, use_continuous_actions = False):

    self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid))

    if use_continuous_actions and not do_interpolation:
      raise Exception("do_interpolation must be true if continuous_actions is true")

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

      # shuffle indicies (so that we update states in a random order reach loop)
      #this *attempts* prevents "circle" bug
      phi_indices = list(range(self.len_phi_grid))
      np.random.shuffle(phi_indices)
      phi_dot_indices = list(range(self.len_phi_dot_grid))
      np.random.shuffle(phi_dot_indices)
      delta_indices = list(range (self.len_delta_grid))
      np.random.shuffle(delta_indices)

      for phi_i in phi_indices:
        for phi_dot_i in phi_dot_indices:
          for delta_i in delta_indices:

            state3_index = (phi_i, phi_dot_i, delta_i)
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)

            if use_continuous_actions:
              state3 = self.state_grid_points[state3_index]
              (_, best_utility) = \
                self.calc_best_action_and_utility_continuous(state3)
            else:
              (_, best_utility) = \
                self.calc_best_action_and_utility(state3_index,do_interpolation)

            #note: utilities of nonfallen states are always positive (and the
            # reward for falling is = 0. then all utilities of valid states will
            # always be greater than the reward for falling)
            reward = self.get_reward(state8)
            if (reward == 0):
              #print("entered top of if statement")
              self.U[state3_index] = 0
            else:
              self.U[state3_index] = reward + gamma*best_utility

      n_episode += 1
      print('Epsiode: ' + str(n_episode))

    print("done trianing, writing csv: " + str(self.Ufile))
    #print(self.U)
    np.savetxt(self.Ufile, self.U.reshape(self.num_states), delimiter = ",")

  def test(self, tmax = 10, state_flag = 0, use_continuous_actions = False,
      gamma = 1, figObject = None, plot_is_inside_last_gridpoint = False):

    # if using continuous actions, need to interpolate value function
    if use_continuous_actions:
      self.itp = RegularGridInterpolator(
          (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
          bounds_error = False, fill_value = 0)

    epsilon = 0; alpha = 0

    reward, time, states8, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, self.reward_flag, True,
        use_continuous_actions, state_flag)

    print("VALUE ITERATION: testing reward: " + str(reward) + ", testing time: "
      + str(time))


    points_inside_last_gridpoint = []
    if plot_is_inside_last_gridpoint:
      points_inside_last_gridpoint = \
        self.calculate_points_inside_last_gridpoint(states8)

    # graph
    figObject = graph.graph(states8, motorCommands, figObject,
      points_inside_last_gridpoint, name = self.Ufile)
    return figObject


  def heatmap_value_function(self):
    #print("phi num = " + str(self.len_phi_grid) + ", phi dot num = "  +
    #  str(self.len_phi_dot_grid) + ", delta num: " + str(self.len_delta_grid))
    #print("U size = " + str(self.U.shape))

    n = 1

    fig1, ax1 = plt.subplots(1,n)
    fig2, ax2 = plt.subplots(1,n)
    fig3, ax3 = plt.subplots(1,n)

    U = self.U
    title = "Value Iteration"

    phi_vs_phidot = np.mean(U, axis = 2)
    #print("phi vs phidot shape: " + str(phi_vs_phidot.shape))

    im1 = ax1.imshow(phi_vs_phidot)
    ax1.set_title("{}: Utility (averaged over delta)".format(title))
    ax1.set_ylabel("phi [rad]")
    ax1.set_xlabel("phi_dot [rad/s]")
    ax1.set_yticks(np.arange(self.len_phi_grid))
    ax1.set_xticks(np.arange(self.len_phi_dot_grid))
    ax1.set_yticklabels(self.phi_grid)
    ax1.set_xticklabels(self.phi_dot_grid)

    fig1.colorbar(im1)

    #figure 2
    phi_vs_delta = np.mean(U, axis = 1)
    #print("phi vs phidot shape: " + str(phi_vs_delta))

    im2 = ax2.imshow(phi_vs_delta)
    ax2.set_title("{}: Utility (averaged over phidot)".format(title))
    ax2.set_ylabel("phi [rad]")
    ax2.set_xlabel("delta [rad]")
    ax2.set_yticks(np.arange(self.len_phi_grid))
    ax2.set_xticks(np.arange(self.len_delta_grid))
    ax2.set_yticklabels(self.phi_grid)
    ax2.set_xticklabels(self.delta_grid)

    fig2.colorbar(im2)

    #figure 3
    phidot_vs_delta = np.mean(U, axis = 0)
    #print("phi vs phidot shape: " + str(phidot_vs_delta))

    im3 = ax3.imshow(phidot_vs_delta)
    ax3.set_title("{}: Utility (averaged over phi)".format(title))
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

  # options = "average", "zero"
  def heatmap_of_policy(self, option, include_linear_controller = False,
    use_continuous_actions = False):

    VI_policy = np.zeros(self.U.shape)
    linear_controller_policy = np.zeros(self.U.shape)

    LQR_controller = LinearController.LinearController()

    if use_continuous_actions:
      self.itp = RegularGridInterpolator(\
        (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
        bounds_error = False, fill_value = 0)

    for phi_i in range(self.len_phi_grid):
      for phi_dot_i in range(self.len_phi_dot_grid):
        for delta_i in range (self.len_delta_grid):

          state3_index = (phi_i, phi_dot_i, delta_i)

          if use_continuous_actions:
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)
            VI_action = self.get_action_continuous(state8, epsilon = 0)
          else:
            action_index = self.act_index(state3_index, epsilon = 0)
            #self.act_index returns which action to take. defined for each model.
            VI_action = self.get_action_from_index(action_index)

          VI_policy[phi_i, phi_dot_i, delta_i] = VI_action

          if include_linear_controller:
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)
            LQR_action = LQR_controller.act(state8)
            #to get shadding to tbe equal, limit max steer rate of linear controller

            if not use_continuous_actions:
              # limit LQR controller to be in the same range as the linear
              # controller. This way the colors are the same on the heatmap
              if LQR_action > self.action_grid[-1]:
                LQR_action = self.action_grid[-1]
              if LQR_action < self.action_grid[0]:
                LQR_action = self.action_grid[0]
            linear_controller_policy[phi_i, phi_dot_i, delta_i] = LQR_action

    if not use_continuous_actions:
      print("Linear Controller output clipped to {} rad/s for \
                  policy heatmap".format(self.action_grid[-1]))

    if include_linear_controller:
      policies = (VI_policy,linear_controller_policy)
    else:
      policies = (VI_policy,)
    n = len(policies)
    titles = ("Value Iteration", "Linear Controller")

    get_middle = lambda array: array[int((len(array)-1)/2)]

    fig1, ax1s = plt.subplots(1,n)
    fig2, ax2s = plt.subplots(1,n)
    fig3, ax3s = plt.subplots(1,n)

    for i in range(n):
      policy = policies[i]
      title = titles[i]

      ax1 = ax1s[i]
      ax2 = ax2s[i]
      ax3 = ax3s[i]

      if option == "average":
        phi_vs_phidot = np.mean(policy, axis = 2)
      elif option == "zero":
        phi_vs_phidot = np.apply_along_axis(get_middle, axis=2, arr= policy)
      #print("phi vs phidot shape: " + str(phi_vs_phidot.shape))

      im1 = ax1.imshow(phi_vs_phidot, cmap=plt.get_cmap("coolwarm"))

      if option == "average":
        ax1.set_title("{} Policy (averaged over steer angles)".format(title))
      elif option == "zero":
        ax1.set_title("{} Policy (with steer angle =0)".format(title))
      ax1.set_ylabel("lean [rad]")
      ax1.set_xlabel("lean rate [rad/s]")
      ax1.set_yticks(np.arange(self.len_phi_grid))
      ax1.set_xticks(np.arange(self.len_phi_dot_grid))
      ax1.set_yticklabels(self.phi_grid)
      ax1.set_xticklabels(self.phi_dot_grid)



      #figure 2


      if option == "average":
        phi_vs_delta = np.mean(policy, axis = 1)
      elif option == "zero":
        phi_vs_delta = np.apply_along_axis(get_middle, axis=1, arr= policy)
      #print("phi vs phidot shape: " + str(phi_vs_delta))

      im2 = ax2.imshow(phi_vs_delta, cmap=plt.get_cmap("coolwarm"))
      if option == "avearge":
        ax2.set_title("{} Policy (averaged over lean rate)".format(title))
      elif option == "zero":
        ax2.set_title("{} Policy (with lean rate =0)".format(title))
      ax2.set_ylabel("lean [rad]")
      ax2.set_xlabel("steer [rad]")
      ax2.set_yticks(np.arange(self.len_phi_grid))
      ax2.set_xticks(np.arange(self.len_delta_grid))
      ax2.set_yticklabels(self.phi_grid)
      ax2.set_xticklabels(self.delta_grid)




      #figure 3

      if option == "average":
        phidot_vs_delta = np.mean(policy, axis = 0)
      elif option == "zero":
        phidot_vs_delta = np.apply_along_axis(get_middle, axis=0, arr= policy)
      #print("phi vs phidot shape: " + str(phidot_vs_delta))

      im3 = ax3.imshow(phidot_vs_delta, cmap=plt.get_cmap("coolwarm"))
      if option == "average":
        ax3.set_title("{} Policy (averaged over lean)".format(title))
      elif option == "zero":
        ax3.set_title("{} Policy (with lean=0)".format(title))
      ax3.set_ylabel("lean rate [rad/s]")
      ax3.set_xlabel("steer [rad]")
      ax3.set_yticks(np.arange(self.len_phi_dot_grid))
      ax3.set_xticks(np.arange(self.len_delta_grid))
      ax3.set_yticklabels(self.phi_dot_grid)
      ax3.set_xticklabels(self.delta_grid)

    cbar1 = fig1.colorbar(im1)
    cbar1.set_label("steer rate [rad/s]")

    cbar2 = fig2.colorbar(im2)
    cbar2.set_label("steer rate [rad/s]")

    cbar3= fig3.colorbar(im3)
    cbar3.set_label("steer rate [rad/s]")

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
