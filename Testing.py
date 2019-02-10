import matplotlib.pyplot as plt
from Qlearning import *
from valueIteration import *
from runBicycleTest import *
import time
import sys

state_flag1 = 3
state_flag2 = 5
figObject = None
simulation_duration = 5

#####################################################3

test_integrators = True


if test_integrators:
  [success, states, figObject] = runBicycleTest(state_flag1,
    controller = LinearController.LinearController(),
    time = simulation_duration , isGraphing  = True, figObject = figObject,
    tstep_multiplier = 1, name = "RK45",
    integrator_method = "RK45")


  [success, states, figObject] = runBicycleTest(state_flag1,
    controller = LinearController.LinearController(),
    time = simulation_duration , isGraphing  = True, figObject = figObject,
    tstep_multiplier = 1, name = "fixed_step_RK4",
    integrator_method = "fixed_step_RK4")


  [success, states, figObject] = runBicycleTest(state_flag1,
    controller = LinearController.LinearController(),
    time = simulation_duration , isGraphing  = True, figObject = figObject,
    tstep_multiplier = 1, name = "Euler",
    integrator_method = "Euler")



plt.show()



plt.close("all")

