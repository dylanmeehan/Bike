import numpy as np
import matplotlib.pyplot as plt
import rhs
import LinearController
import graph
from unpackState import *
from tableBased import *

# use Euler Integration to simulate a bicycle
def runBicycleTest(stateflag = 4, controller = LinearController.LinearController(),
  time = 10, isGraphing  = True, figObject = None):

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
    u = controller.act(state)

    zdot = rhs.rhs(state,u)

    #update state. Euler Integration
    prevState = state
    state = state + zdot*timestep

    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    #check if bike has fallen
    if (np.abs(phi) >= np.pi/4):
      print("Bike has fallen; Test Failure\n")
      success = False
      break

    states[count,:] = state
    motorCommands[count] = u


    count = count + 1

  figObject = graph.graph(states, motorCommands, figObject)

  if isGraphing:
    plt.show()
    plt.close("all")

  return([success, states, figObject])

#runBicycleTest()