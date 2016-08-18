function [y] = BraninFunction(x)
%BraninFunction This is the Branin function. Let's optimize it.

y = (x(2) - (5.1/(4*pi^2))*x(1)^2 + (5/pi)*x(1) - 6)^2 + 10*(1 - 1/(8*pi))*cos(x(1)) + 10;
y = -y; % We want to maximize the function, so we invert it.

end