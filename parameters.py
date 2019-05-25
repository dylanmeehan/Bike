# properties of the p struct in matlab
G = 9.81 # acceleration due to gravity
L = 1.02 #length of wheel base
B = 0.33 #distance from rear wheel to COM projected onto ground
H = 0.516 # height of COM in point mass model. see ABT fall 2017 report.
C = 0.0 # trail is zero

MAX_STEER_RATE = 4.8 #rad/s See ABT fall 2017 report
delta_threshold = 1.4
FALLING_THRESHOLD = 0.785398


print("PARAMETERS: using max steer rate: ", MAX_STEER_RATE,
 " , and steer angle limit: ", delta_threshold,
 "\n falling threshold: ", FALLING_THRESHOLD)
