import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *

class Qlearning(TableBased):

  def __init__(self, state_grid_flag, action_grid_flag ):
    super(Qlearning, self).__init__(state_grid_flag, action_grid_flag)
    self.Q = np.zeros((self.len_phi_grid,self.len_phi_dot_grid,\
      self.len_delta_grid, self.num_actions))

    k = 11 #number of basis functions. TODO: remove hardcode
    W = np.zeros((self.num_actions, k))

  #given state3_index.
  # epsilon: determines the probability of taking a random action
  # return: the action to take (if epsilon=0, this is the "best" action)
  def act_index(self, state_grid_point_index, epsilon):
    #pick a random action sometimes
    if np.random.random() < epsilon:
        a = np.random.randint(0, self.num_actions)
        # return a random index in the action space
        return a

    #pick the action with the highest Q value
    return np.argmax(self.Q[state_grid_point_index])
      #returns the index of the maximum value of the 1D array Q[state_index]
      #that index corresponds to a specific action

  #Qlearning update rule
  #updates Q table (in place, updateQ does not return anything)
  def update_Q(self, state3_index, reward, a_index, state3_next_index, done,
    alpha, gamma):
    max_q_next = np.max(
      self.Q[state3_next_index[0],state3_next_index[1],state3_next_index[2],:])

    #don't update Q if the simulation has terminated
    self.Q[state3_index[0],state3_index[1],state3_index[2],a_index] +=\
          alpha * (1.0-done) * (reward+gamma*max_q_next-\
            self.Q[state3_index[0],state3_index[1],state3_index[2],a_index])

  def toBasisFunctions(self, state3):
  [phi, phi_dot, delta] = state3

  f = np.array([1, phi, phi_dot, phi**2, phi_dot**2, phi*phi_dot,
      delta, delta**2, delta*phi, phi*delta**2, phi**2*delta])
  return f

  def update_W(self, state3_index, reward, a_index, state3_n)

  # for Function Approximation of Q table
  def Qvalue_fapprox(self, state3, action_index):
    weights = W[action_index, :] #get weights for only one action
    basis_fuctions = self.toBasisFunctions(state3)
    Q = np.dot(weights, basis_functions)

  def act_index_fapprox(self, state3, epsilon):
    #duplicated code
    #pick a random action sometimes
    if np.random.random() < epsilon:
        a = np.random.randint(0, self.num_actions)
        # return a random index in the action space
        return a

    return np.argmax(self.Qvalue_fapprox(state3, self.action_grid))
    #can I call the function on a list of action_Grid. OR, do I need
    # to vectorize the function/ use anonymous functions to do this

  # saves Q table in Q.csv
  def train(self, epsilon = 1, epsilon_decay = 0.9998, epsilon_min = 0.05,
    gamma = 1, alpha = 0.5, alpha_decay = 0.9998, alpha_min = 0.1,
    num_epsiodes = 7000, tmax = 10, state_flag = 0):

    reward_history = []
    reward_averaged = []
    time_history = []
    time_averaged = []
    alphas = []
    epsilons = []

    for episode in range(num_epsiodes):

      if epsilon > epsilon_min:
        epsilon *= epsilon_decay

      if alpha > alpha_min:
        alpha *= alpha_decay

      total_reward, total_time, _, _ = \
        self.simulate_episode(epsilon, gamma, alpha, tmax, False, state_flag)

      reward_history.append(total_reward)
      average_reward50 = np.average(reward_history[-50:])
      reward_averaged.append(average_reward50)

      time_history.append(total_time)
      average_time50 = np.average(time_history[-50:])
      time_averaged.append(average_time50)


      if (episode%500 == 0):
        print("episode:" + str(episode) + ", reward:" + str(average_reward50)\
          +", time: " + str(average_time50))

      alphas.append(alpha)
      epsilons.append(epsilon)

    #end of for loop

    np.savetxt("Q.csv", \
      self.Q.reshape((self.num_states, self.num_actions)), delimiter=",")

    print("avg time at end:" + str(average_time50))
    print("num episodes:" + str(episode))

    t = range(episode+1)
    fig, ax1 = plt.subplots()
    ax1.plot(t,reward_averaged)
    ax2 = ax1.twinx()
    ax2.plot(t,alphas)
    ax2.plot(t,epsilons)
    ax2.legend(["alpha", "epsilon"])
    plt.show()


  def test(self, Qfile = "Q2.csv", tmax = 10, state_flag = 0, gamma = 1,
    figObject = None):

    savedQ = np.genfromtxt(Qfile, delimiter=',')
    self.Q = savedQ.reshape((self.len_phi_grid, self.len_phi_dot_grid, \
      self.len_delta_grid, self.num_actions))

    epsilon = 0
    alpha = 0

    reward, time, states8, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, True, state_flag)

    print("Q LEARNING: testing reward: " + str(reward) + ", testing time: " +
      str(time))

    figureObjects = graph.graph(states8, motorCommands, figObject)
    return figureObjects



#Qlearning_model = Qlearning(state_grid_flag = 0, action_grid_flag = 0)
# #Qlearning_model.train()
# #print(Qlearning_model.Q)

#figObject = Qlearning_model.test(Qfile = "Q.csv", tmax = 10, state_flag = 0,
#   gamma =1, figObject = None)
# #I don't need to store the figObject returned by test. this returns the same
# # figObject as before. We only get a new FigObject when we initialize figObject
# # to None
# runBicycleTest(stateflag = 4,
#   controller = LinearController.LinearController(), time = 10, isGraphing  = True):
# Qlearning_model.test(Qfile = "Q.csv", tmax = 10, state_flag = 0,
#   gamma =1, figObject = figObject)

#plt.show()
#plt.close("all")

