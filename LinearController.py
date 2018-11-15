import numpy as np
from parameters import *
from unpackState import *

class LinearController(object):

  # "Arundathi" Gains: 71, 21, -20
  # LQR gains: 24, 7, -8

  def __init__(self, k1 = 24., k2 = 7., k3 = -8.):
    self.k1 = k1;
    self.k2 = k2;
    self.k3 = k3;

  def act(self, state):
    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    #control variable
    u = self.k1*phi + self.k2*phi_dot + self.k3*delta

    #clip u to within maximum steer rate
    if u > MAX_STEER_RATE:
      u = MAX_STEER_RATE
    elif u < -MAX_STEER_RATE:
      u = -MAX_STEER_RATE


    return(u)