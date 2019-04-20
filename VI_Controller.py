from ControllerClass import *
from scipy.interpolate import RegularGridInterpolator
from valueIteration.py import *


class VI_Controller(ValueIteration, Controller):
  def __init__(self, use_continuous_actions = False,
    use_continuous_state_with_discrete_actions = False,
    controller_integration_method = "fixed_step_RK4",
    use_regression_model_of_table = False):

    self.use_continuous_actions = use_continuous_actions
    self.use_continuous_state_with_discrete_actions = use_continuous_state_with_discrete_actions
    self.controller_integration_method = controller_integration_method

    self.itp = RegularGridInterpolator(
    (self.phi_grid, self.phi_dot_grid, self.delta_grid),self.U,
    bounds_error = False, fill_value = 0, method = "linear")

    #use_regression_of_table determines if we use a regression model to replace
    # the table. If true, instead of looking up stuff in the table, we use the
    # regression model to predict the values of states
    self.use_regression_model_of_table = use_regression_model_of_table
    if use_regression_model_of_table:
      self.run_regression()


