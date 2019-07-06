close all

phi_dot = linspace(-2,2,101);

g = 9.81;
h = 0.516;

phi = acos(1 -h/(2*g).*phi_dot.^2) .* -sign(phi_dot);
disp(1 -h/(2*g).*phi_dot.^2)
phi_linear = -sqrt(h/g).*phi_dot;

figure
plot(phi_dot, phi);
hold on
plot(phi_dot, phi_linear);
legend("nonlinear", "linear")
shg

percent_error = abs(phi-phi_linear)/abs(phi)*100;
max_percent_error = percent_error(end)