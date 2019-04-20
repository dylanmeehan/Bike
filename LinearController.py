import numpy as np
from parameters import *
from unpackState import *
from ControllerClass import *

class LinearController(Controller):

  # both arundathi and LQR gains work on the real bicycle
  # "Arundathi" Gains: 71, 21, -20
  # LQR gains from Spring 2018: 24, 7, -8

  def __init__(self, k1 = 24., k2 = 7., k3 = -8.):
    self.k1 = k1;
    self.k2 = k2;
    self.k3 = k3;

    self.is_initialized = True

  def act(self, state, timestep):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    #control variable
    u = self.k1*phi + self.k2*phi_dot + self.k3*delta

    #clip u to within maximum steer rate
    if u > MAX_STEER_RATE:
      u = MAX_STEER_RATE
    elif u < -MAX_STEER_RATE:
      u = -MAX_STEER_RATE


    return(u)