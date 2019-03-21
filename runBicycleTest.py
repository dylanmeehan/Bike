import numpy as np
import matplotlib.pyplot as plt
import rhs
import LinearController
import graph
from unpackState import *
from tableBased import *
import integrator
import scipy.integrate as inter

# use Euler Integration to simulate a bicycle
def runBicycleTest(stateflag = 4, controller = LinearController.LinearController(),
  reward_flag = 8, time = 10, isGraphing  = True, figObject = None, tstep_multiplier = 1,
  name = "LQR", test_rh45_uncontrolled = False, integrator_method = "Euler",
  USE_LINEAR_EOM = False, timestep = 1/50):

  print("running test with LQR controller")

  state = getStartingState8(stateflag)

  numTimeSteps = int(time/timestep)+1

  #create arrays before loop
  success = True
  numStates = state.size
  states = np.zeros([numTimeSteps, numStates])
  motorCommands = np.zeros([numTimeSteps, 1])

  #initialize starting values of arrays
  states[1,:] = state
  motorCommands[1] = 0

  #used for comparision of rk45 integrator to Euler Integrator, not genearlly used.
  if test_rh45_uncontrolled:
    rhs_fun = lambda t,state: rhs.rhs(state,0, USE_LINEAR_EOM = False)

    #give list of times at which to get solution
    tspan = list(np.linspace(0, time, num=time/timestep+1))

    #solve ode (uncontrolled)
    solution = inter.solve_ivp(rhs_fun, [0, time], state, method='RK45', t_eval = tspan,
      rtol = 1e-12, atol = 1e-12)
    states = solution.y
    states = states.T
    #print(states)

    #find count to truncate values after the bike has fallen
    phis = states[:,3]
    count1 = np.argmax(phis>np.pi/4)
    count2 = np.argmax(phis<-np.pi/4)
    count = max(count1, count2)


  else:
    count = 0
    cum_reward = 0
    sim_time = 0

    while( count < numTimeSteps):

      #calculate control action
      u = controller.act(state)

      new_state, reward, is_done = step(state, u, reward_flag, tstep_multiplier = 1,
        method = "fixed_step_RK4", USE_LINEAR_EOM = USE_LINEAR_EOM, timestep = timestep)

      states[count,:] = state
      motorCommands[count] = u

      sim_time += timestep
      cum_reward += reward
      count += 1

      state = new_state

  states = states[:count,:]
  motorCommands = motorCommands[:count]

  print(name + ": score:" + str(cum_reward) + ", testing time: "
      + str(sim_time))

  figObject = graph.graph(states, motorCommands, figObject, [], name)

  return([success, states, figObject])

#runBicycleTest()