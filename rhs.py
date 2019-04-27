from parameters import *
from unpackState import *
import numpy as np
from numba import jit

#@jit(nopython=True)
def rhs(state, u, USE_LINEAR_EOM):
    # Equivalent to rhs in Matlab. (as of 10/27/18)
    # Calculates the derivative of the state variables

    [t, x, y, phi, psi, delta, phi_dot, v] = unpackState(state)

    #calculate derivates
    tdot = 1.0 #include time?
    xdot = v * np.cos(psi)
    ydot = v * np.sin(psi)
    phi_dot = phi_dot #in some reports, phi_dot = w_r
    psi_dot = (v/L)*(np.tan(delta)/np.cos(phi))
    delta_dot = u # ideal steer rate
    v_dot = 0.0
    if USE_LINEAR_EOM:
      phi_ddot = (((-(v**2))*delta) - B*v*u + G*L*phi)/(H*L)
    else:
      phi_ddot =( (1/H)*
                (G*np.sin(phi) - np.tan(delta)
                    *((v**2)/L + B*v_dot/L + np.tan(phi)
                        *((B/L)*v*phi_dot
                        -(H/(L**2)*(v**2)*np.tan(delta))))
                -B*v*delta_dot/(L*np.cos(delta)**2))
            )
    # Returns u which is the motor command and the zdot vector in the form of a list
    zdot = np.array([tdot, xdot, ydot, phi_dot, psi_dot, delta_dot, phi_ddot, v_dot])
    #print(zdot)
    #print(zdot.dtype)
    return(zdot)