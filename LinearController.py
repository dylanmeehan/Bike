import numpy as np
from parameters import *
from unpackState import *
from ControllerClass import *

def getLQRGains(name):
  LQRlibrary = {
    "sp18":[24.,7.,-8.],
    "lqrd_2m_s": [31.6875, 7.2238, -5.5283],
    "lqrd_3m_s": [23.859, 5.418, -7.556],
    "lqrd_1m_s": [67.113, 15.371, -2.792],
    "lqrd_0.5m_s": [ 161.63995, 37.062367, -0.42609913],
    "lqrd_0.25m_s": [ 369.00165, 84.624927, 1.2895433],
  }

  return LQRlibrary[name]

class LinearController(Controller):

  # both arundathi and LQR gains work on the real bicycle
  # "Arundathi" Gains: 71, 21, -20
  # LQR gains from Spring 2018: 24, 7, -8

  def __init__(self,K = [24., 7., -8]):
    self.k1 = K[0];
    self.k2 = K[1];
    self.k3 = K[2];

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