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