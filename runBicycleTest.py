import numpy as np
import matplotlib.pyplot as plt
import rhs
import LinearController
import graph
from unpackState import *
from tableBased import *
import integrator

# use Euler Integration to simulate a bicycle
def runBicycleTest(stateflag = 4, controller = LinearController.LinearController(),
  time = 10, isGraphing  = True, figObject = None, tstep_multiplier = 1,  name = "LQR"):

  print("running test")

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

  count = 0;
  while( count < numTimeSteps):

    #calculate control action
    u = 0 #controller.act(state)

    # integrate the odes
    state = integrator.integrate(state, u, timestep, tstep_multiplier)

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