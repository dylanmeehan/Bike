import numpy as np


class StateGridPoints:

  #discritize each of the state variables. Construct self.state_grid_points
  # which is a meshgrid of these points
  #given: state_grid_flag determines which grid points to use
  def set_state_grid_points(self, state_grid_flag):
    print("Using State Grid Flag " + str(state_grid_flag))

    def make_full_grid(half_grid):
      return [-1*i for i in half_grid[::-1]] + [0] + half_grid

    #generate grid in which to discritize states

    if state_grid_flag == 0:
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1]
      self.phi_dot_grid = make_full_grid(phi_dot_halfgrid)
      delta_half_grid =   [.02, .05, .1, .2, .3, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)

    #DON'T USE FOR VECTORIZED TESTING. .8 phi is past falling lolz
    #19x17x15 states
    elif state_grid_flag == 1:
      #self.phi_grid = [-.8, -.6, -.4, -.28, -.22, -.16, -.1, -.06, -.02, 0, \
      #  .02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .8 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, .2, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)


    #small state space for testing timing (9x7x7)
    elif state_grid_flag == 2:
      phi_half_grid = [.02, .06, .1, .16 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, ]
      self.delta_grid = make_full_grid(delta_half_grid)

   #small state space for testing timing (11x9x7)
    elif state_grid_flag == 3:
      phi_half_grid = [.02, .06, .1, .16, .22 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1]
      self.delta_grid = make_full_grid(delta_half_grid)

   #19x17x15 states, all states are not fallen states lol
    elif state_grid_flag == 4:
      #.785 is falling criteria
      phi_half_grid = [.02, .06, .1, .16, .22, .28, .4, .6, .77 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .05, .1, .2, .3, .4,  .7, 1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  [.02, .05, .1, .2, .4,  .7, 1]
      self.delta_grid = make_full_grid(delta_half_grid)


    elif state_grid_flag == 5:
      #.785 is falling criteria
      phi_half_grid = [.02, .04, .07, .10, .13, .16, .20, .25, .3, .35, .4, .45,
        .5, .55, .6, .65, .7, .77 ]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .04, .07, .1, .15, .2, .25,.3,.4,.5,.6,.7,.8,.9,1 ]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.02, .04, .07, .1, .15, .2, .25,.3,.4,.5,.75,1 ]
      self.delta_grid = make_full_grid(delta_half_grid)

    #tight grid space, but only cloose to the balanced state
    # 27x 29 x 25 state space
    elif state_grid_flag == 6:
      #.785 is falling criteria
      phi_half_grid = [.02, .04, .06, .08, .10, .12, .14, .16, .18, .20, .22, .24, .26]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .04, .06, .08, .10, .13, .17, .2, .25, .3, .35, .4, .45, .5]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.02, .04, .06, .08, .10, .13, .16, .19, .22, .25, .28, .31]
      self.delta_grid = make_full_grid(delta_half_grid)

     #tight grid space, same as 6 but increase phi dot so that se don't overrun
    # 27x 29 x 25 state space
    elif state_grid_flag == 7:
      #.785 is falling criteria
      phi_half_grid = [.02, .04, .06, .08, .10, .12, .14, .16, .18, .20, .22, .24, .26]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .04, .06, .08, .10, .13, .17, .2, .25, .3, .35, .4, .45,
       .5, 0.55, 0.6, 0.65, 0.7]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.02, .04, .06, .08, .10, .13, .16, .19, .22, .25, .28, .31]
      self.delta_grid = make_full_grid(delta_half_grid)

     #tight grid space, same as 6 but increase phi dot so that se don't overrun
    # 27x 49 x 35 state space
    elif state_grid_flag == 8:
      #.785 is falling criteria
      phi_half_grid = [.02, .04, .06, .08, .10, .12, .14, .16, .18, .20, .22, .24, .26]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .04, .06, .08, .10, .13, .17, .2, .25, .3, .35, .4, .45,
       .5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.02, .04, .06, .08, .10, .13, .16, .19, .22, .25, .28, .31,
      .35, .39, .43, .47, .51]
      self.delta_grid = make_full_grid(delta_half_grid)

    elif state_grid_flag == 11:
      phi_half_grid = list(np.linspace(0.01,0.26,26, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)


    elif state_grid_flag == 12:
      phi_half_grid = list(np.linspace(0.001,0.009,9, endpoint=True)) + \
      list(np.linspace(0.01,0.26,26, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)

    elif state_grid_flag == 14:
      phi_half_grid = list(np.linspace(0.001,0.009,9, endpoint=True)) + \
        list(np.linspace(0.01,0.26,26, endpoint=True)) + \
        list(np.linspace(0.28,0.78,26, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid = list(np.linspace(0.02,1,50, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)

    elif state_grid_flag == 15:
      phi_half_grid = list(np.linspace(0.001,0.009,9, endpoint=True)) + \
        list(np.linspace(0.01,0.26,26, endpoint=True)) + \
        list(np.linspace(0.28,0.78,26, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = list(np.linspace(0.02,1,50, endpoint=True)) +\
        list(np.linspace(1.04,5,100, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid = list(np.linspace(0.02,1,50, endpoint=True)) + \
        list(np.linspace(1.04,2,25, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)

    elif state_grid_flag == 16:
      phi_half_grid = list(np.linspace(0.01,0.27,13, endpoint=True)) + \
        list(np.linspace(0.30,0.78,13, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = list(np.linspace(0.02,0.5,25, endpoint=True)) +\
        list(np.linspace(.55,2.5,25, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid = list(np.linspace(0.02,0.50,25, endpoint=True)) + \
        list(np.linspace(0.55,1.5,20, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)


    elif state_grid_flag == 13:
      phi_half_grid = list(np.linspace(0.001,0.009,9, endpoint=True)) + \
      list(np.linspace(0.01,0.26,26, endpoint=True))
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid =  list(np.linspace(0.002,0.018,9, endpoint=True)) + \
      list(np.linspace(0.02,1,50, endpoint=True))
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =  list(np.linspace(0.002,0.018,9, endpoint=True)) + \
      list(np.linspace(0.02,1,50, endpoint=True))
      self.delta_grid = make_full_grid(delta_half_grid)

    #small grid, half the states as state_grid_flag 9 in each direction
    # 15 x 25 x 19 state space
    elif state_grid_flag == 9:
      #.785 is falling criteria
      phi_half_grid = [.02, .06,  .10, .14, .18, .22, .26]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.02, .06, .10, .17, .25, .35, .45, 0.55, 0.65, 0.75,
      0.85, 0.95]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.02, .06, .10, .16, .22, .28, .35, .43, .51]
      self.delta_grid = make_full_grid(delta_half_grid)

      # same size as 8 but arranged differently. more logrithmically
    elif state_grid_flag == 10:
      #.785 is falling criteria
      phi_half_grid = [.01, .02, .03, .04, .05, .07, .09, .11, .14, .17, .2, .23, .26]
      self.phi_grid = make_full_grid(phi_half_grid)
      phi_dot_half_grid = [.01, .02, .03, .05, .07, .09, .12, .16, .2, .25, .3, .35, .4, .45,
       .5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
      self.phi_dot_grid = make_full_grid(phi_dot_half_grid)
      delta_half_grid =   [.01, .02, .03, .05, .07, .09, .12, .15, .18, .22, .26, .3,
      .34, .38, .42, .46, .51]
      self.delta_grid = make_full_grid(delta_half_grid)


    else:
      raise Exception("Invalid state_grid_flag: {}".format(state_grid_flag))

    # calculate lengths once and store their values
    self.len_phi_grid = len(self.phi_grid)
    self.len_phi_dot_grid = len(self.phi_dot_grid)
    self.len_delta_grid = len(self.delta_grid)
    self.num_states = self.len_phi_grid*self.len_phi_dot_grid*self.len_delta_grid

    #used for checking if we are within one grid point of the goal state
    self.smallest_phi = phi_half_grid[0]
    self.smallest_phi_dot = phi_dot_half_grid[0]
    self.smallest_delta = delta_half_grid[0]

    # generate a 3D grid of the points in our table
    # a mesh where each point is a 3 tuple. one dimension for each state variable
    phi_points, phi_dot_points,  delta_points  = \
      np.meshgrid(self.phi_grid, self.phi_dot_grid, self.delta_grid, indexing='ij')
    self.state_grid_points = np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')