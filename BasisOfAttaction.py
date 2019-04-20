
def plot_basis_of_attraction(controller, state_table_flag):

  GridPoints = StateGridPoints
  GridPoints.set_state_grid_points(state_table_flag)

  phi_points, phi_dot_points,  delta_points = \
    np.meshgrid(self.phi_grid, self.phi_dot_grid, [0])
  phi_and_phi_dot_points= np.rec.fromarrays([phi_points, phi_dot_points,
      delta_points], names='phi_points,phi_dot_points,delta_points')

  print(phi_and_phi_dot_points)