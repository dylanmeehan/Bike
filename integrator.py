import rhs
import scipy

#solve the EoMs numerically
def integrate(state8, u, controller_timestep, tstep_multiplier = 1):

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

  return state8