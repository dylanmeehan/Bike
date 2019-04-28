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



      #add in -1 to match sign convention change between phi for Shihao's and Dylan's derivations
    #   phi_ddot_dylan = -1*(     (G*np.sin(-phi))/H
    #    - (v**2*np.sin(-phi))/(L**2*np.cos(-phi))
    #    + (v**2*np.sin(-phi))/(L**2*np.cos(delta)**2*np.cos(-phi))
    #    + (v**2*np.sin(delta))/(H*L*np.cos(delta)) + (B*delta_dot*v)/(H*L*np.cos(delta)**2)
    #     + (B*v_dot*np.sin(delta))/(H*L*np.cos(delta)) +
    #     (B*-phi_dot*v*np.sin(delta)*np.sin(-phi))/(H*L*np.cos(delta)*np.cos(-phi)) )
    #   #sometmes phi_dot is a list of 1 element. wtf? so convert it to a signal scalar

    #   # phi_ddot_dylan = np.asscalar(phi_ddot_dylan)
    #   # #print("phi_ddot: {:f}".format(phi_ddot))
    #   # #print("phi_ddot dylan: {:f}".format(phi_ddot_dylan))
    #   # assert(np.abs(phi_ddot - phi_ddot_dylan) < 1e-6)
    #   # phi_ddot = phi_ddot_dylan

    # if isinstance(phi_ddot,list):
    #   print("phi_ddot " + str(phi_ddot) + " is a list. ????")
    # if isinstance(phi_ddot,list):
    #   print("phi_ddot_dylan " + str(phi_ddot_dylan) + " is a list. ????")
    # assert(np.abs(phi_ddot - phi_ddot_dylan) < 1e-6)
    #phi_ddot = np.asscalar(phi_ddot)
    # Returns u which is t(g*sin(phi))/h - (v^2*sin(phi))/(l^2*cos(phi)) + (v^2*sin(phi))/(l^2*cos(delta)^2*cos(phi)) + (v^2*sin(delta))/(h*l*cos(delta)) + (b*delta_dot*v)/(h*l*cos(delta)^2) + (b*v_dot*sin(delta))/(h*l*cos(delta)) + (b*phi_dot*v*sin(delta)*sin(phi))/(h*l*cos(delta)*cos(phi))he motor command and the zdot vector in the form of a list

    zdot = np.array([tdot, xdot, ydot, phi_dot, psi_dot, delta_dot, phi_ddot, v_dot])
    # for element in zdot:
    #     print(type(element))
    #print(zdot)
    #print(zdot.dtype)
    return(zdot)