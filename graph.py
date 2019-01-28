import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from unpackState import *

#set default font size
mpl.rcParams['font.size']=14

#graph variables about a bicycle simulation
def graph(states, motorCommands, figObject,  points_inside_last_gridpoint = [],
  name = ""):

  print("graphing: " + name)

  #if there is not already some graphs, make new graphs
  if figObject == None:
    figObject = [plt.subplots(2,2), plt.subplots(2,1)]

  [ts, xs, ys, phis, psis, deltas, phi_dots, vs] =  \
    np.apply_along_axis(unpackState, 1, states).T

  (fig1, [[ax1,ax2],[ax3,ax4]]),(fig2, [ax5,ax6]) = figObject

  #fig1, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
  ax1.plot(ts, phis, label=name)
  ax1.legend(loc="center right")
  ax1.set_title("lean vs time")
  ax1.set_xlabel("time [s]")
  ax1.set_ylabel("lean angle [rad]")

  ax2.plot(ts, phi_dots)
  ax2.set_title("lean rate vs time")
  ax2.set_xlabel("time [s]")
  ax2.set_ylabel("lean rate [rad/s]")

  ax3.plot(ts, deltas)
  ax3.set_title("steer vs time")
  ax3.set_xlabel("time [s]")
  ax3.set_ylabel("steer angle [rad]")

  ax4.plot(ts, motorCommands)
  ax4.set_title("steer rate (u) vs time")
  ax4.set_xlabel("time [s]")
  ax4.set_ylabel("steer rate [rad/s]")

  # # print out phi at a set time to compare integrators
  # #account for off-by-one error
  # if name == "Single rk45":
  #   n = 23
  # else:
  #   n = 22
  # print(str(name)+ ": phi at time " + str(ts[n]) + " is " + str(phis[n]))


  if points_inside_last_gridpoint != []:
    for ax in [ax1, ax2, ax3, ax4]:
      ax0 = ax.twinx()
      ax0.plot(ts, points_inside_last_gridpoint, color = 'y')
      ax0.set_ylabel("Is controller inside last gridpoint")

  #fig2, [ax5,ax6] = plt.subplots(2,1)
  ax5.plot(xs, ys, label=name)
  ax5.legend(loc="upper left")
  ax5.set_title("trajectory")
  ax5.set_xlabel("x position [m]")
  ax5.set_ylabel("y position [m]")
  ax5.axis('equal')

  psi_dots = np.diff(psis)
  ax6.plot(ts[0:-1], psi_dots)
  ax6.set_title("yaw rate vs time")
  ax6.set_xlabel("time [s]")
  ax6.set_ylabel("yaw rate [rad/s]")

  fig1.tight_layout()
  fig2.tight_layout()
  #plt.show()
  # plt.show() waits until you close the figure


  return (figObject)
