import matplotlib.pyplot as plt
import numpy as np
from unpackState import *

#graph variables about a bicycle simulation
def graph(states, motorCommands):
  [ts, xs, ys, phis, psis, deltas, phi_dots, vs] =  \
    np.apply_along_axis(unpackState, 1, states).T


  fig1, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
  ax1.plot(ts, phis)
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

  fig2, [ax5,ax6] = plt.subplots(2,1)
  ax5.plot(xs, ys)
  ax5.set_title("trajectory")
  ax5.set_xlabel("x position [m]")
  ax5.set_ylabel("y position [m]")
  ax5.axis('equal')

  psi_dots = np.diff(psis)
  ax6.plot(ts[0:-1], psi_dots)
  ax6.set_title("yaw rate vs time")
  ax6.set_xlabel("time [s]")
  ax6.set_ylabel("yaw rate [rad/s]")

  plt.show()

  plt.close('all')