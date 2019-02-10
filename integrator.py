import rhs
import scipy
import scipy.integrate as inter
from unpackState import *

#solve the EoMs numerically
# method = ["Euler", "RK45", "fixed_step_RK4"]
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

  elif method == "fixed_step_RK4":
    #print("using fixed_step_RK4 method")
    dt = controller_timestep    #use same time step as controller
    f = lambda s: rhs.rhs(s,u)  #f = ydot = rhs
    y0 = state8                 # IC, initial state, y0

    k1 = dt*f(y0)
    k2 = dt*f(y0 + 1/2*k1)
    k3 = dt*f(y0 + 1/2*k2)
    k4 = dt*f(y0 + k3)

    state8 = y0 + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4  #do integration
    #state8 = y1 = state value at final timestep.

  elif method == "RK45":
    #calculate time to integrate
    [t_start, x, y, phi, psi, delta, phi_dot, v] = unpackState(state8)
    t_end = t_start + controller_timestep
    #print("integrating stuff at t=" + str(t_start))

    rhs_fun = lambda t,state: rhs.rhs(state,u)

    tspan = list(np.linspace(t_start, t_end, 10))

    #solve ode (uncontrolled)
    solution = inter.solve_ivp(rhs_fun, [t_start, t_end], state8, method='RK45')
    states8 = solution.y
    states8 = states8.T
    state8 = states8[-1,:]


  else:
    raise ValueError("invalid method: "+str(method))

  return state8