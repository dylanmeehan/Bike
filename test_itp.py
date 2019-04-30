import numpy as np
from scipy.interpolate import RegularGridInterpolator

phi_grid = np.array([-1, 0, 1, 2])
phi_dot_grid = np.array([0,1,2])
delta_grid = np.array([0,1])

U = np.array([[ [1, 0,1,2],[4, 3,4,5],[7, 6,7,8]],
  [ [9, 6, 9, 4],[10, 13, 10, 7 ],[8,9,8,6]]]).T
#print(np.shape(U))

itp = RegularGridInterpolator((phi_grid, phi_dot_grid, delta_grid),U,
  bounds_error = False, fill_value = 0, method = "linear")

test_points = np.array([
[0,0,0],
[1,1,0.5],
[1,0.5,0.5],
[0,0,-1],
[.5,.5,.5],
[.25,.5,.5],
[.75,.5,.5]
  ])

result_points = itp(test_points)
print("results: ",result_points)
correct_results = np.array([
  0,
  7,
  6,
  0,
  5.75,
  5.625,
  5.875
  ])


all_correct = np.equal(result_points,  correct_results).all()
print("all_correct: ", all_correct)