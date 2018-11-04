import numpy as np

def unpackState(state):
  t = state[0];
  x= state[1];
  y= state[2];
  phi= state[3];
  psi= state[4];
  delta= state[5];
  phi_dot= state[6];
  v =  state[7];

  return(np.array([t, x, y, phi, psi, delta, phi_dot, v]))

#turn a 3 state into a state
def state3_to_state(state3, v = 3):
  phi = state3[0]
  phi_dot = state3[1]
  delta = state3[2]

  x = 0; y = 0; t = 0; psi = 0;

  return [t, x, y, phi, psi, delta, phi_dot, v]


#turn a state into a state3
def state_to_state3(state):
  [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

  #state3 is a tuple
  return (phi, phi_dot, delta)


#def unpackStates(states):
#  [ts, xs, ys, phis, psis, deltas, phi_dots, vs] = \
#    np.apply_along_axis(unpackState, 1, states).T