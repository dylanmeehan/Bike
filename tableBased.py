import numpy as np
import rhs
import integrator
import matplotlib.pyplot as plt
import graph
from unpackState import *
from scipy.interpolate import interpn
import time
from pathlib import Path

class TableBased(object):

  timestep = 1/50

  # create points for actions based on action_grid_flag
  # store actions in action_grid
  def set_action_grid_points(self, action_grid_flag):

    if action_grid_flag == 0:
      self.action_grid = [-2, -1, 0, 1, 2]
    elif action_grid_flag == 1:
      self.action_grid = np.linspace(-2,2,11, endpoint=True)
    elif action_grid_flag == 2:
      self.action_grid = np.linspace(-2,2,101, endpoint=True)
    elif action_grid_flag == 3: #lol, improved my ability to count
      self.action_grid = np.linspace(-2,2,81, endpoint=True)
    elif action_grid_flag == 4:
      # Flag for continuous action space
      self.action_grid = None

    else:
      raise Exception("Invalid action_grid_flag: {}".format(action_grid_flag))

    if action_grid_flag == 4:
      self.num_actions = np.inf
    else:
      self.num_actions = len(self.action_grid)


  #discritize each of the state variables. Construct self.state_grid_points
  # which is a meshgrid of these points
  #given: state_grid_flag determines which grid points to use
  def set_state_grid_points(self, state_grid_flag):

    def make_full_grid(half_grid):
      return [-1*i for i in half_grid[::-1]] + [0] + half_grid

    #generate grid in which to discritize states

    if state_grid_flag == 0:
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1]
      self.phi_dot_grid = make_full_grid(phi_dot_halfgrid)
      delta_half_grid =   [.02, .05, .1, .2, .3, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)

    #DON'T USE FOR VECTORIZED TESTING. .8 phi is past falling lolz
    #19x17x15 states
    elif state_grid_flag == 1:
      #self.phi_grid = [-.8, -.6, -.4, -.28, -.22, -.16, -.1, -.06, -.02, 0, \
      #  .02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, .2, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)


    #small state space for testing timing (9x7x7)
    elif state_grid_flag == 2:
      phi_half_grid = [.02, .06, .1, .16 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, ]
      self.delta_grid = make_full_grid(delta_half_grid)

   #small state space for testing timing (11x9x7)
    elif state_grid_flag == 3:
      phi_half_grid = [.02, .06, .1, .16, .22 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1]
      self.delta_grid = make_full_grid(delta_half_grid)

   #19x17x15 states, all states are not fallen states lol
    elif state_grid_flag == 4:
      #.785 is falling criteria
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .77 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, .2, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)


    else:
      raise Exception("Invalid state_grid_flag: {}".format(state_grid_flag))

    # calculate lengths once and store their values
    self.len_phi_grid = len(self.phi_grid)
    self.len_phi_dot_grid = len(self.phi_dot_grid)
    self.len_delta_grid = len(self.delta_grid)
    self.num_states = self.len_phi_grid*self.len_phi_dot_grid*self.len_delta_grid

    #used for checking if we are within one grid point of the goal state
    self.smallest_phi = phi_half_grid[0]
    self.smallest_phi_dot = phi_dot_half_grid[0]
    self.smallest_delta = delta_half_grid[0]

    # generate a 3D grid of the points in our table
    # a mesh where each point is a 3 tuple. one dimension for each state variable
    phi_points, phi_dot_points,  delta_points  = \
      np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing='ij')
    self.state_grid_points = np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  def __init__(self, state_grid_flag, action_grid_flag, reward_flag):
    self.set_state_grid_points(state_grid_flag)
    self.set_action_grid_points(action_grid_flag)
    self.reward_flag = reward_flag

  #given: phi_index, phi_dot_index, delta_index
  #returns:continous, full - 8 variable, state for this index
  def state8_from_indicies(self, phi_index, phi_dot_index, delta_index):
    state3_index = (phi_index, phi_dot_index, delta_index)
    state3 = self.state_grid_points[state3_index]
    return state3_to_state8(state3)

  def state3_index_to_state3(self, state3_index):
    return self.state_grid_points[state3_index]

  #given: index of action in action_grid
  # return: action, (continous valued) steer rate command
  def get_action_from_index(self, action_index):
    return self.action_grid[action_index]

  # return the 3-tuple of the indicies of the state grid point closest to state8
  #return a state3 variable
  def discretize(self, state8):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    closest_phi_point = np.abs(phi-self.phi_grid).argmin()
    closest_phi_dot_point = np.abs(phi_dot-self.phi_dot_grid).argmin()
    closest_delta_point = np.abs(delta-self.delta_grid).argmin()

    return (closest_phi_point, closest_phi_dot_point, closest_delta_point)

  #this function only works for states which are the state gridpoints.
  #this is useful for value Iteration
  #step table maps state indicies and action indicies to the next state
  def setup_step_table(self, reward_flag, step_table_integration_method = "fixed_step_RK4"):
    #t0 = time.time()

    if Path(self.step_table_file).is_file():
      print("Loading step_table {} from file".format(self.Ufile))

      saved_step_table = np.genfromtxt(self.step_table_file, delimiter = ",")
      self.step_table = saved_step_table.reshape(self.len_phi_grid,
        self.len_phi_dot_grid, self.len_delta_grid, self.num_actions,3)

      saved_reward_table = np.genfromtxt(self.reward_file, delimiter = ",")
      self.reward_table = saved_reward_table.reshape(self.len_phi_grid,
        self.len_phi_dot_grid, self.len_delta_grid)



    else:
      print("Making new step table {}".format(self.Ufile))
      self.step_table = np.zeros((self.len_phi_grid, self.len_phi_dot_grid,
        self.len_delta_grid, self.num_actions, 3))
      self.reward_table = np.zeros((self.len_phi_grid, self.len_phi_dot_grid,
        self.len_delta_grid))
        #8 is number of variables in a continuous state

      for i_phi in range(self.len_phi_grid):
        for i_phi_dot in range(self.len_phi_dot_grid):
          for i_delta in range(self.len_delta_grid):

            state8 = self.state8_from_indicies(i_phi, i_phi_dot, i_delta)
              #don't use reward from stepping because that is reward for next state
              # s', and not current state, s (I think)
            s_reward = self.get_reward(state8, reward_flag)
            self.reward_table[i_phi, i_phi_dot, i_delta] = s_reward

            for i_action in range(self.num_actions):

              action = self.action_grid[i_action]

              new_state8, next_s_reward, _ = self.step(state8, action, reward_flag,
                method = step_table_integration_method)
              new_state3 = state8_to_state3(new_state8)

              self.step_table[i_phi, i_phi_dot, i_delta, i_action] = new_state3

      np.savetxt(self.step_table_file,
        self.step_table.reshape(self.num_states*self.num_actions*3), delimiter = ",")
      np.savetxt(self.reward_file,
        self.reward_table.reshape(self.num_states), delimiter = ",")

      #





    #t1 = time.time()
    #print("Setup step_table (precalculated state transitions) in " + str(t1-t0)
    #  + "seconds")

    #print("shape of reward table:" + str(np.shape(self.reward_table)))

  #given: a 3-tuple state3_index of the indicies for phi, phi_dot, delta
  #       the index of the action to take
  # return: state (continous, 8 varible) corresponding to taking the action
  # at action index in the state represented by state3_index
  def step_fast(self, state3_index, action_index):
    phi_i = state3_index[0]
    phi_dot_i = state3_index[1]
    delta_i = state3_index[2]

    return self.step_table[phi_i, phi_dot_i, delta_i, action_index]

  #given: state (a state in the continous space)
  #       u (an action in the continoue state)
  # tstep_multiplier is the number of integration_timesteps for every one
  # controller timestep. Should be an integer >= 1
  #return: (state, reward, isDone)
  def step(self, state8, u, reward_flag, tstep_multiplier = 1,  method = "Euler"):

    state8 = integrator.integrate(state8, u, self.timestep, tstep_multiplier = 1,
      method = method)

    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    # check if bike has fallen, only need to do once a timestep
    if (np.abs(phi) >= np.pi/4):
      #print("Bike has fallen; Test Failure")
      isDone = True
    else:
      isDone = False

    reward = self.get_reward(state8)

    return (state8, reward, isDone)

  #given: state8 - the state to get the reward of
  #       reward_flag - dictates what reward shaping to use
  # returns: reward - a nunber greater than or equal to  0
  # reward = 0 iff the bike has fallen (phi > pi/4)
  def get_reward(self,state8, reward_flag = 3):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

    #REWARD_FOR_FALLING = 0
    #Do not change (if the bike falls, the utility
    # of that state is set to 0, ie ,no need to look at utilities going forward
    # so, the reward_for_falling should always be 0 and all other rewards should
    # be greater than 0, so when the bicycle does not fall, it gets a utility
    # greater than that if it falls).

    # test ifbike has fallen
    if (abs(phi) > np.pi/4):
      reward = 0
      #If reward is changed from 0, also change fill_value in the interpolator,
      #I should set them both to be dependent on 1 parameter
    else:
      if reward_flag == 0:
        reward =  1 #no shapping
      elif reward_flag == 1:
        reward = (1-(abs(phi))/2 - np.sign(phi)*phi_dot/20) #basic reward shaping
      elif reward_flag == 2:
        reward = 1/(phi**2+0.01) #add a little bit, so that we don't divide by 0
      #garantees reward for not falling down is greater than that for falling
      elif reward_flag == 3:
        reward = 5 - phi**2
      elif reward_flag == 4:
        reward = 3 - np.abs(phi)
      else:
        raise Exception("Invalid reward_flag: {}".format(reward_flag))

      assert (reward > 0)

    return reward

  # do 1 simulation of the bicycle
  # state_flag determines the starting state
  # can be used to test or for training a Qlearning agent
  def simulate_episode(self, epsilon, gamma, alpha, tmax, reward_flag,
    isTesting, use_continuous_actions, state_flag = 0, integration_method = "Euler"):

    state8 = getStartingState8(state_flag)

    is_done = False
    total_reward = 0
    total_time = 0

    # return index of closest point to phi, phi_dot, and delta
    state_grid_point_index = self.discretize(state8)

    maxNumTimeSteps = int(tmax/self.timestep)+1

    if not use_continuous_actions:
      # return index of closest point to phi, phi_dot, and delta
      state_grid_point_index = self.discretize(state8)

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
    while( (count < maxNumTimeSteps) and (not is_done)):

      if use_continuous_actions:
        action = self.get_action_continuous(state8, epsilon,
         integration_method = integration_method)
        #print("continuous action:" + str(action))
      else:
        action_index = self.act_index(state_grid_point_index, epsilon)
        #self.act_index returns which action to take. defined for each model.
        action = self.get_action_from_index(action_index)
        #print("discrete action:" + str(action))


      new_state8, reward, is_done = self.step(state8, action, reward_flag,
        method = integration_method )

      if not use_continuous_actions:
        new_state_grid_point_index = self.discretize(new_state8)

      if (not isTesting):
        self.update_Q(state_grid_point_index, reward, action_index, \
          new_state_grid_point_index, is_done, alpha, gamma)

      total_reward += reward
      if (not is_done):
        total_time += self.timestep

      state8 = new_state8

      #print state3 values
      #print([state8[3],state8[5], state8[6]])

      if not use_continuous_actions:
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
