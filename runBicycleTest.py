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
  time = 10, isGraphing  = True, figObject = None, tstep_multiplier = 1,  name = "LQR",
  test_rh45_uncontrolled = False, integrator_method = "Euler"):

  print("running test with LQR controller")

  state = getStartingState8(stateflag)

  timestep = 1/50 #seconds
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
    rhs_fun = lambda t,state: rhs.rhs(state,0)

    #give list of times at which to get solution
    tspan = list(np.linspace(0, time, num=time/timestep+1))

    #solve ode (uncontrolled)
    solution = inter.solve_ivp(rhs_fun, [0, time], state, method='RK23', t_eval = tspan,
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
    count = 0;
    while( count < numTimeSteps):

      #calculate control action
      u = controller.act(state)

      # integrate the odes
      state = integrator.integrate(state, u, timestep, tstep_multiplier,
        method = integrator_method)

      [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

      #check if bike has fallen
      if (np.abs(phi) >= np.pi/4):
        print("Bike has fallen; Test Failure\n")
        success = False
        break

      states[count,:] = state
      motorCommands[count] = u


      count = count + 1

  states = states[:count,:]
  motorCommands = motorCommands[:count]

  figObject = graph.graph(states, motorCommands, figObject, [], name)

  return([success, states, figObject])

#runBicycleTest()