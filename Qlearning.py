import numpy as np
import rhs
import matplotlib.pyplot as plt
import graph
from unpackState import *
from tableBased import *

class Qlearning(TableBased):

  def __init__(self ):
    super(Qlearning, self).__init__()
    self.Q = np.zeros((self.len_phi_grid,self.len_phi_dot_grid,\
      self.len_delta_grid, self.num_actions))

  #given state_grid_point_index. A 3-tuple. (a discritized state)
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

  def update_Q(self, s_indicies, reward, a_index, s_next_indicies, done, alpha, gamma):
    max_q_next = np.max(
      self.Q[s_next_indicies[0],s_next_indicies[1],s_next_indicies[2],:])

    #don't update Q if the simulation has terminated
    self.Q[s_indicies[0],s_indicies[1],s_indicies[2],a_index] +=\
          alpha * (1.0-done) * (reward+gamma*max_q_next-\
            self.Q[s_indicies[0],s_indicies[1],s_indicies[2],a_index])

  def train(self, epsilon = 1, epsilon_decay = 0.9998, epsilon_min = 0.05, \
    gamma = 1, alpha = 0.5, alpha_decay = 0.9998, alpha_min = 0.1, \
    num_epsiodes = 500, tmax = 10, state_flag = 0):

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

    print("avg reward(time) at end:" + str(average_reward50))
    print("num episodes:" + str(episode))

    t = range(episode+1)
    fig, ax1 = plt.subplots()
    ax1.plot(t,reward_averaged)
    ax2 = ax1.twinx()
    ax2.plot(t,alphas)
    ax2.plot(t,epsilons)
    ax2.legend(["alpha", "epsilon"])
    plt.show()


  def test(self, Qfile = "Q.csv", tmax = 10, state_flag = 0, gamma = 1):
    savedQ = np.genfromtxt(Qfile, delimiter=',')
    self.Q = savedQ.reshape((self.len_phi_grid, self.len_phi_dot_grid, \
      self.len_delta_grid, self.num_actions))

    epsilon = 0
    alpha = 0

    reward, time, states, motorCommands = \
      self.simulate_episode(epsilon, gamma, alpha, tmax, True, state_flag)

    print("testing reward: " + str(reward) + ", testing time: " + str(time))
    graph.graph(states, motorCommands)



Qlearning_model = Qlearning()
Qlearning_model.train()
#print(Qlearning_model.Q)

Qlearning_model.test(Qfile = "Q.csv", tmax = 10, state_flag = 0, gamma =1)