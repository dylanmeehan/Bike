import numpy as np

def unpackState(state8):
  assert len(state8) == 8
  t = state8[0];
  x= state8[1];
  y= state8[2];
  phi= state8[3];
  psi= state8[4];
  delta= state8[5];
  phi_dot= state8[6];
  v =  state8[7];

  return(np.array([t, x, y, phi, psi, delta, phi_dot, v]))

def state3_to_state8(state3, v = 3):

  assert len(state3) == 3
  phi = state3[0]
  phi_dot = state3[1]
  delta = state3[2]

  x = 0; y = 0; t = 0; psi = 0;

  return [t, x, y, phi, psi, delta, phi_dot, v]

#getStartingState returns an 8 value continuous state.
#this is not affected by the discritization grid used for the table methods
def getStartingState8(state_flag = 0):
  starting_states = {
    0: np.array([0, 0, 0, 0.01, 0, 0, 0, 3]),
    1: np.array([0, 0, 0, np.pi/32, 0, 0, 0, 3]),
    2: np.array([0, 0, 0, np.random.uniform(-np.pi/16, np.pi/16) , 0, 0, 0, 3]),
    3: np.array([0, 0, 0, np.pi/16, 0, 0, 0, 3]),
    4: np.array([0, 0, 0, np.pi/8, 0, 0, 0, 3]),
    5: np.array([0, 0, 0, -np.pi/16, 0, 0, 0, 3]),
    6: np.array([0, 0, 0, -np.pi/32, 0, 0, 0, 3]),
    7: np.array([0, 0, 0, -0.77, 0, 0, 0, 3]),
    8.1: np.array([0, 0, 0, -0.78, 0, 0, 2.5, 3]),
    8.2: np.array([0, 0, 0, 0.78, 0, 0, 2.5, 3]),
    8.3: np.array([0, 0, 0, -0.78, 0, 0, -2.5, 3]),
    8.4: np.array([0, 0, 0, 0.78, 0, 0, -2.5, 3])
  }
  return starting_states[state_flag]

def state8_to_state3(state8):
  assert len(state8) == 8
  [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)

  #state3 is a tuple
  return (phi, phi_dot, delta)

#def unpackStates(states):
#  [ts, xs, ys, phis, psis, deltas, phi_dots, vs] = \
#    np.apply_along_axis(unpackState, 1, states).T