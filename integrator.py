import rhs
import scipy
import scipy.integrate as inter
from unpackState import *

#solve the EoMs numerically
# method = ["Euler", "RK45"]
def integrate(state8, u, controller_timestep, tstep_multiplier = 1,
  method = "Euler"):

  if method == "Euler":

    #self.timestep is the timestep of the controller (say 1/50 s)
    #integration_timestep is the timestep of the integrator
    integration_timestep = controller_timestep/tstep_multiplier

    #loop through the euler integrator tstep_multiplier number of times
    # u is constant in this loop but the state changes. That is, a single controller
    # output is used for the whole timestep even though their may be multiple
    # integration timesteps
    count = 0
    for _ in range(tstep_multiplier):
      # take in a state and an action
      zdot = rhs.rhs(state8,u)

      #update state. Euler Integration
      #prevState8 = state8
      state8 = state8 + zdot*integration_timestep
      #print(str(state8))
      count += 1

  elif method == "RK45":
    #calculate time to integrate
    [t_start, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)
    t_end = t_start + controller_timestep
    print("integrating stuff at t=" + str(t_start))

    rhs_fun = lambda t,state: rhs.rhs(state,u)

    tspan = list(np.linspace(t_start, t_end, 10))
    #solve ode (uncontrolled)
    solution = inter.solve_ivp(rhs_fun, [t_start, t_end], state8, method='RK45')
    states8 = solution.y
    states8 = states8.T
    print(states8)
    state8 = states8[-1,:]
    #print(state8)
    print(np.shape(state8))
    #state8 = state8.T
    #print(states)

  else:
    raise ValueError("invalid method: "+str(method))

  return state8