import numpy as np
import rhs
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import graph
import scipy
import parameters as params
from unpackState import *
from tableBased import *
from scipy.interpolate import RegularGridInterpolator

from pathlib import Path
import scipy.optimize as opt
import LinearController
import time

#set default font size
mpl.rcParams['font.size']=14

class ValueIteration(TableBased):

  #initializing makes table for rewards and state transitions, so if any of those
  # are changed after the original initialization, that will cause a problem.
  # remake_table remakes both step table and reward table
  def __init__(self, state_grid_flag, action_grid_flag, reward_flag,
    Ufile = "modelsB/valueIteration_U.csv", use_only_continuous_actions = False,
    step_table_integration_method = "fixed_step_RK4", remake_table = False,
    USE_LINEAR_EOM = False, name = None):

    print("Initializing VI model")
    init_t1 = time.time()

    super(ValueIteration, self).__init__(state_grid_flag, action_grid_flag,
      reward_flag, USE_LINEAR_EOM)

    self.step_table_file = Ufile+ "_step_table.csv"
    self.reward_file = Ufile+ "_reward_table.csv"
    self.Ufile = Ufile + ".csv"
    self.name = name

    #to be used in regression
    self.map_to_basis_functions = lambda phi, phi_dot, delta: [
      1, phi, phi_dot, delta, phi**2, phi_dot**2, delta**2, phi*phi_dot,
      phi_dot*delta, phi*delta]

    if USE_LINEAR_EOM:
      print(name + ": ********* USE_LINEAR_EOM = " + str(USE_LINEAR_EOM) + " *********")
   # self.step_file

    if not use_only_continuous_actions:
      self.setup_step_table(reward_flag, remake_table, step_table_integration_method)



    if Path(self.Ufile).is_file() and not remake_table:
      saved_U = np.genfromtxt(self.Ufile, delimiter = ",")
      self.U = saved_U.reshape(self.len_phi_grid, self.len_phi_dot_grid,
          self.len_delta_grid)
    else:
      self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid,
        self.len_delta_grid))
      print("Creating new VI file " + Ufile)



    init_t2 = time.time()
    print("Initialized VI Model " + self.Ufile + " in " + str(init_t2-init_t1) + "sec")
    #TODO: precompute reward function. Set up table (similiar to step_table), so
    #       that we don't compute the reward for a state every time. We only
    #       look up the reward

  #given: state3_index: the index of a point in the descritized table
  #     do_intepolations: boolean to decide to interpolate or not
  #return: (best_action_index, best_action_utility) for that state.
  # best_action_index is the index of the action which has the highest utility
  # best_action_utility is the utility of that action
  def calc_best_action_and_utility(self, state3_index, gamma,
    make_action_utility_graph = False):

    Qtemp = np.zeros(self.num_actions)

    state3 = self.state_grid_points[state3_index]

    for action_index in range(self.num_actions):

      new_state3 = self.step_fast(state3_index, action_index)
      reward = self.reward_table[state3_index[0], state3_index[1],
          state3_index[2], action_index]

      #new_state3 = state8_to_state3(new_state8)
      #TODO: change step_fast to use lookup table which returns new_state3

      # reward = R(s,a)
      # Q(s,a) = R(s,a) + gamma * U(s')
      Qtemp[action_index] = reward + gamma*self.get_value(new_state3)

    if make_action_utility_graph:
      self.graph_action_vs_utility(action_index, Qtemp)

    best_action_utility = np.max(Qtemp)
    best_action_index = np.argmax(Qtemp)

    return (best_action_index, best_action_utility)

  def calc_best_action_and_utility_continuous_state(self, state8, gamma,
    integration_method = "fixed_step_RK4", use_regression = False):

    Qtemp = np.zeros(self.num_actions)

    for action_index in range(self.num_actions):
      action = self.action_grid[action_index] #equivalent to get_action_from_index()


      (new_state8, reward, _) = self.step(state8, action, self.reward_flag,
       method = integration_method)

      #new_state3 = state8_to_state3(new_state8)
      #TODO: change step_fast to use lookup table which returns new_state3

      # reward = R(s,a)
      # Q(s,a) = R(s,a) + gamma * U(s')

      if use_regression:
        phi= new_state8[3];
        delta= new_state8[5];
        phi_dot= new_state8[6];

        current_value = np.dot(self.map_to_basis_functions(phi, phi_dot, delta),
          self.regression_coefficients)
        Qtemp[action_index] = reward + gamma*current_value
      else:
        Qtemp[action_index] = reward + gamma*self.get_value(state8_to_state3(new_state8))

    best_action_utility = np.max(Qtemp)
    best_action_index = np.argmax(Qtemp)


    return (best_action_index, best_action_utility)

  def continuous_utility_function(self, state8, u, integration_method, gamma,
    use_regression = False):

    (new_state8, reward, isDone) = self.step(state8, u, self.reward_flag,
      method = integration_method)

    new_state3 = state8_to_state3(new_state8)

    if use_regression:
      phi= new_state8[3];
      delta= new_state8[5];
      phi_dot= new_state8[6];

      current_value = np.dot(self.map_to_basis_functions(phi, phi_dot, delta),
        self.regression_coefficients)
      utility = reward + gamma*current_value
    else:
      utility = reward + gamma*self.get_value(new_state3)
    #reward = R(s,a). utility of s' comes exclusively from get_value(s')
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
  def calc_best_action_and_utility_continuous_action(self, state8,
    integration_method = "Euler", gamma = 1, use_regression = False):

    #we need to minimiza something. minimizning negations of the utility function
    # is equivalent to maximizing the utilty function
    negated_utility_fun = lambda u: -1*self.continuous_utility_function(state8, u,
      integration_method = integration_method, gamma = gamma,
      use_regression = use_regression)

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

    utility_of_best_action = self.continuous_utility_function(state8, u,
      integration_method, gamma)
    #print(u)

    return (u, utility_of_best_action)

  def get_action_continuous(self, state8, epsilon = 0, gamma = 1,
    integration_method = "Euler", use_regression = False):
    (u, _) = self.calc_best_action_and_utility_continuous_action(state8,
      integration_method = integration_method, gamma = gamma,
      use_regression = use_regression)
    return u

  # given: state3_index (a discritized state 3 tuple).
  # return: the index of the best action to take
  def act_index(self, state3_index, epsilon, gamma,
    make_action_utility_graph = False, time = None):

    (best_action_index, _) = self.calc_best_action_and_utility(state3_index, gamma,
      make_action_utility_graph = make_action_utility_graph)
    return best_action_index

  # return a value for the (continuous) new_state3
  def get_value(self,new_state3):

    #the interpolator (itp) is initialized in train, and interpolation_method
    # sets the interpolator to use either linear or nearest methods
    return self.itp([new_state3[0],new_state3[1],new_state3[2]])


  #trains a valueIteration, table-based mode.
  #when training finishes, utilities are stored in a csv
  # if continuous actions is true, do_interpolation must be true
  # vectorize == True updates states using vectorized code (fast)
  def train(self, gamma = 1, num_episodes = 30,
    interpolation_method = "linear", use_continuous_actions = False, vectorize = None):

    if vectorize == None:
      vectorize = not use_continuous_actions

    train_t1 = time.time()

    self.U = np.zeros((self.len_phi_grid,self.len_phi_dot_grid, self.len_delta_grid))

    if use_continuous_actions and interpolation_method != "linear":
      raise Exception("'interpolation_method' must be 'linear' if 'continuous_actions' is 'true'")

    n_episode = 0

    while (n_episode < num_episodes):
      tstart = time.time()



      self.itp = RegularGridInterpolator(\
        (self.phi_grid, self.phi_dot_grid, self.delta_grid), self.U,
        bounds_error = False, fill_value = 0, method = interpolation_method)
        # false bounds error causes us to extrapolate values out of range
        #fill_value = 0, sets the value outside of the interpolation range to 0
        # thus, if the bicycle gets no reward for a state outside of the grid
        # this ensures bad states have a reward of 0 (as desired)
        #t4 = time.time()

      # shuffle indicies (so that we update states in a random order reach loop)
      #this *attempts* prevents "circle" bug
      phi_indices = list(range(self.len_phi_grid))
      #np.random.shuffle(phi_indices)
      phi_dot_indices = list(range(self.len_phi_dot_grid))
      #np.random.shuffle(phi_dot_indices)
      delta_indices = list(range (self.len_delta_grid))
      #np.random.shuffle(delta_indices)
      action_indicies = list(range(self.num_actions))


      def update_state(state3_index):
        phi_i = state3_index[0]; phi_dot_i = state3_index[1]; delta_i = state3_index[2]
        state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)

        if use_continuous_actions:
          state3 = self.state_grid_points[state3_index]
          state8 = state3_to_state8(state3)
          (_, best_utility) = \
            self.calc_best_action_and_utility_continuous_action(state8, gamma = gamma)
            #gamma is inherited from outer class

        else:
          (_, best_utility) = \
            self.calc_best_action_and_utility(state3_index, gamma)

        self.U[state3_index] = best_utility
        #print("calc_best_action_and_utility in " + str(t_2-t_1) + "sec")

        #note: utilities of nonfallen states are always positive (and the
        # reward for falling is = 0. then all utilities of valid states will
        # always be greater than the reward for falling)
        #this if statement is unecessary. All states in the grid should not be
        #fallend states. Otherwise, they are useless to have in the grid.
        #reward is for the current state, not the state we would get to.
        #additionally, any state outside of the grid would get assigned a value
        # of 0 by the interpolator, so any potentially fallen states would have
        # a value of 0
        #print("calculated reward in : " + str(t_4-t_3) + " sec")

      #assume we are not using continuous actions, but are doing interpolation
      #let indicies_matrix be states and actions
      def update_state_vectorized(indicies_matrix):

        #lookup 4 tuple in step lookup table
        lookup = lambda indicies: \
          self.step_table[indicies[0], indicies[1], indicies[2], indicies[3]]
        new_states = np.apply_along_axis(lookup, 0, indicies_matrix)
        #print("new_states shape is " + str(np.shape(new_states)))


        #t_1 = time.time()
        #interpolate values
        value_of_states_and_actions = np.transpose(self.itp(new_states.T))
        #print("value_of_states_and_actions shape is " + str(np.shape(value_of_states_and_actions)))

        #find max value for each state3. Don't do this because U
        #value_of_states = np.amax(value_of_states_and_actions, axis = 3)
        #print("value_of_states shape is " + str(np.shape(value_of_states)))

        #Q values contain both states and actions: Q(s,a)
        #reward value depends on (s,a)
       # print("reward shape: " + str(np.shape(self.reward_table)))
       # print("value_of_states_and_actions shape:" + str(np.shape(value_of_states_and_actions)))
        Q_values = self.reward_table + gamma*value_of_states_and_actions
        #U(s) = max_a Q(S,a)
        self.U =  np.amax(Q_values, axis = 3)

        #t_2 = time.time()
        #print("calc_best_action_and_utility in " + str(t_2-t_1) + "sec")

       # t_3 = time.time()
        #note: utilities of nonfallen states are always positive (and the
        # reward for falling is = 0. then all utilities of valid states will
        # always be greater than the reward for falling)
        # reward_matricies = self.get_reward(state8)
        # if (reward == 0):
        #   #print("entered top of if statement")
        #   self.U[state3_index] = 0
        # else:
        #   self.U[state3_index] = reward + gamma*best_utility
        # t_4 = time.time()
        # #print("calculated reward in : " + str(t_4-t_3) + " sec")


      if vectorize:
        indicies_matrix = np.meshgrid(phi_indices, phi_dot_indices, delta_indices,
        action_indicies, indexing = "ij")
        #print("indicies_matrix shape is: " + str(np.shape(indicies_matrix)))

        update_state_vectorized(indicies_matrix)

      else:
        for phi_i in phi_indices:
          for phi_dot_i in phi_dot_indices:
            for delta_i in delta_indices:

              state3_index = (phi_i, phi_dot_i, delta_i)
              update_state(state3_index)

      tloop = time.time()
      #for phi_i in phi_indices:
      #  for phi_dot_i in phi_dot_indices:
      #    for delta_i in delta_indices:



      n_episode += 1
      tend = time.time()
      #print("For loop in " + str(tend-tloop) + " sec")
      print('Epsiode: ' + str(n_episode) + " in " + str(tend-tstart) + " sec")

    print("done trianing, writing csv: " + str(self.Ufile))
    #print(self.U)
    np.savetxt(self.Ufile, self.U.reshape(self.num_states), delimiter = ",")


    train_t2 = time.time()
    print("Trained VI Model " + self.Ufile + " in " + str(train_t2-train_t1) + "sec")


  # integration_method = "fixed_step_RK4", "Euler"
  def test(self, tmax = 10, state_flag = 0, use_continuous_actions = False,
      use_continuous_state_with_discrete_actions = True,
      gamma = 1, figObject = None, plot_is_inside_last_gridpoint = False,
      integration_method = "fixed_step_RK4", name = None, use_regression = False,
      timesteps_to_graph_actions_vs_utilites = []):

    if use_regression:
      self.run_regression()

    if name == None:
      name = self.Ufile

    t_test1 = time.time()

    self.itp = RegularGridInterpolator(
        (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
        bounds_error = False, fill_value = 0, method = "linear")

    epsilon = 0; alpha = 0

    reward, time_testing, states8, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, self.reward_flag, True,
        use_continuous_actions, use_continuous_state_with_discrete_actions,
        state_flag, integration_method = integration_method,
        use_regression = use_regression, timesteps_to_graph_actions_vs_utilites =
        timesteps_to_graph_actions_vs_utilites)

    print("VALUE ITERATION: testing reward: " + str(reward) + ", testing time: "
      + str(time_testing))


    points_inside_last_gridpoint = []
    if plot_is_inside_last_gridpoint:
      points_inside_last_gridpoint = \
        self.calculate_points_inside_last_gridpoint(states8)

    # graph
    figObject = graph.graph(states8, motorCommands, figObject,
      points_inside_last_gridpoint, name)

    t_test2 = time.time()
    print("Tested VI Model " + self.Ufile + " in " + str(t_test2-t_test1))

    return figObject


  # option = ["average", "zero"]
  def heatmap_value_function(self, option, use_regression = False):
    #print("phi num = " + str(self.len_phi_grid) + ", phi dot num = "  +
    #  str(self.len_phi_dot_grid) + ", delta num: " + str(self.len_delta_grid))
    #print("U size = " + str(self.U.shape))

    get_middle = lambda array: array[int((len(array)-1)/2)]

    print("making heatmap of value function")

    n = 1

    fig1, ax1 = plt.subplots(1,n)
    fig2, ax2 = plt.subplots(1,n)
    fig3, ax3 = plt.subplots(1,n)

    if use_regression:
      (phi_coords, phi_dot_coords, delta_coords) = \
        np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing="ij")

      get_value_from_regression = lambda phi, phi_dot, delta : \
        np.dot(self.map_to_basis_functions(phi, phi_dot, delta),self.regression_coefficients)

      U = np.asarray(list(map(get_value_from_regression, phi_coords,
        phi_dot_coords, delta_coords)))
    else:
      U = self.U


    title = "Value Iteration"


    if option == "average":
      phi_vs_phidot = np.mean(U, axis = 2)
    elif option == "zero":
      phi_vs_phidot = np.apply_along_axis(get_middle, axis=2, arr= U)
    else:
      raise("Invalid option for heatmap_value_function")

    #print("phi vs phidot shape: " + str(phi_vs_phidot.shape))

    im1 = ax1.imshow(phi_vs_phidot)
    ax1.set_title("{}: Utility: {} delta".format(title,option))
    ax1.set_ylabel("phi [rad]")
    ax1.set_xlabel("phi_dot [rad/s]")
    ax1.set_yticks(np.arange(self.len_phi_grid))
    ax1.set_xticks(np.arange(self.len_phi_dot_grid))
    ax1.set_yticklabels(self.phi_grid)
    ax1.set_xticklabels(self.phi_dot_grid)

    fig1.colorbar(im1)

    #figure 2
    if option == "average":
      phi_vs_delta = np.mean(U, axis = 1)
    elif option == "zero":
      phi_vs_delta = np.apply_along_axis(get_middle, axis=1, arr= U)
    else:
      raise("Invalid option for heatmap_value_function")
    #print("phi vs phidot shape: " + str(phi_vs_delta))

    im2 = ax2.imshow(phi_vs_delta)
    ax2.set_title("{}: Utility: {} phidot".format(title,option))
    ax2.set_ylabel("phi [rad]")
    ax2.set_xlabel("delta [rad]")
    ax2.set_yticks(np.arange(self.len_phi_grid))
    ax2.set_xticks(np.arange(self.len_delta_grid))
    ax2.set_yticklabels(self.phi_grid)
    ax2.set_xticklabels(self.delta_grid)

    fig2.colorbar(im2)

    #figure 3
    if option == "average":
      phidot_vs_delta = np.mean(U, axis = 0)
    elif option == "zero":
      phidot_vs_delta = np.apply_along_axis(get_middle, axis=0, arr= U)
    else:
      raise("Invalid option for heatmap_value_function")

    #print("phi vs phidot shape: " + str(phidot_vs_delta))

    im3 = ax3.imshow(phidot_vs_delta)
    ax3.set_title("{}: Utility: {} phi".format(title,option))
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
    use_continuous_actions = False, linear_controller = LinearController.LinearController()):

    print("making heatmap of policy")
    if use_continuous_actions:
      print("*** using continuous actions may cause problems for the solver \
      for using continuous actions ")

    VI_policy = np.zeros(self.U.shape)
    linear_controller_policy = np.zeros(self.U.shape)

    self.itp = RegularGridInterpolator(\
      (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
      bounds_error = False, fill_value = 0)

    for phi_i in range(self.len_phi_grid):
      for phi_dot_i in range(self.len_phi_dot_grid):
        for delta_i in range (self.len_delta_grid):

          state3_index = (phi_i, phi_dot_i, delta_i)

          if use_continuous_actions:
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)
            VI_action = self.get_action_continuous(state8, epsilon = 0, gamma = 1)
          else:
            action_index = self.act_index(state3_index, epsilon = 0, gamma = 1)
            #self.act_index returns which action to take. defined for each model.
            VI_action = self.get_action_from_index(action_index)

          VI_policy[phi_i, phi_dot_i, delta_i] = VI_action

          if include_linear_controller:
            state8 = self.state8_from_indicies(phi_i, phi_dot_i, delta_i)
            linear_action = linear_controller.act(state8)
            #to get shadding to tbe equal, limit max steer rate of linear controller

            if not use_continuous_actions:
              # limit LQR controller to be in the same range as the linear
              # controller. This way the colors are the same on the heatmap
              if linear_action > self.action_grid[-1]:
                linear_action = self.action_grid[-1]
              if linear_action < self.action_grid[0]:
                linear_action = self.action_grid[0]
            linear_controller_policy[phi_i, phi_dot_i, delta_i] = linear_action

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


  def plot_level_sets(self):
    mean = np.mean(self.U, axis=None)
    std = np.std(self.U, axis=None)
    dmax = np.max(self.U, axis=None)
    dmin = np.min(self.U, axis=None)

    print("mean: " + str(mean))
    print("std: " + str(std))
    print("dmax: " + str(dmax))
    print("dmin: " + str(dmin))

    print("size of U: " + str(np.shape(self.U)))

    (phi_coords, phi_dot_coords, delta_coords) = \
     np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing="ij")
    print("size of phi coords: " + str(np.shape(phi_coords)))
    print("size of phi_dot coords: " + str(np.shape(phi_dot_coords)))
    print("size of delta coords: " + str(np.shape(delta_coords)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #set points violating the first condition to be NaN, so they are not plotted,
    #otherwise, do not change their value


    subset_delta_coords = np.where((self.U <125) & (self.U>120), delta_coords, np.nan)
    ax.scatter(phi_coords, phi_dot_coords, subset_delta_coords, c = "r")

    subset_delta_coords = np.where((self.U >149.5) & (self.U<149.6), delta_coords, np.nan)
    ax.scatter(phi_coords, phi_dot_coords, subset_delta_coords, c = "c")

    subset_delta_coords = np.where((self.U <149.9) & (self.U>149.85), delta_coords, np.nan)
    ax.scatter(phi_coords, phi_dot_coords, subset_delta_coords, c = "b")


    subset_delta_coords = np.where((self.U>149.99), delta_coords, np.nan)
    ax.scatter(phi_coords, phi_dot_coords, subset_delta_coords, c = "g")

    ax.legend(["120<U<125", "149.5<149.6",  "149.85<U<149.9", "U>149.99"])

    ax.set_zlim(self.delta_grid[0], self.delta_grid[-1])
    ax.set_ylim(self.phi_dot_grid[0], self.phi_dot_grid[-1])
    ax.set_xlim(self.phi_grid[0], self.phi_grid[-1])

    ax.set_xlabel('phi')
    ax.set_ylabel('phi_dot')
    ax.set_zlabel('delta')



    plt.show()

  def run_regression(self):

    print("running regression")

    U = self.U.flatten()
    # print("size of U" + str(np.shape(U)))

    (phi_coords, phi_dot_coords, delta_coords) = \
     np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing="ij")


    phi_coords = phi_coords.flatten()
    phi_dot_coords = phi_dot_coords.flatten()
    delta_coords = delta_coords.flatten()
    # print("size of phi coords: " + str(np.shape(phi_coords)))
    # print("size of phi_dot coords: " + str(np.shape(phi_dot_coords)))
    # print("size of delta coords: " + str(np.shape(delta_coords)))

    inputs = np.asarray(list(map(self.map_to_basis_functions, phi_coords,
      phi_dot_coords, delta_coords)))
    # print("size of inputs: " + str(np.shape(inputs)))

    reg = LinearRegression().fit(inputs, U)
    print("reg score:" + str(reg.score(inputs, U)))

    self.regression_coefficients = reg.coef_
    print("reg coeffs: " + str(self.regression_coefficients))

  def graph_action_vs_utility(self, action_index, Qtemp):
    actions = np.zeros(self.num_actions)
    for i in list(range(self.num_actions)):
      actions[i] = self.get_action_from_index(i)

    fig, ax = plt.subplots()
    ax.plot(actions, Qtemp)

    ax.set(xlabel='action', ylabel='value',
       title='action vs value ')




