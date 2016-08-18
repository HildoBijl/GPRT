function [ y ] = trialFunction(x)
%trialFunction This is a trial function used to test the algorithm on. It is basically a semi-random sum of sines and cosines. The input vector x has to be a vertical vector of size two, or a
%collection of such vectors.

if size(x,1) ~= 2
	error('The input vector x should satisfy size(x,1) == 2. It currently does not. Make sure it is a vector of size two or a set of such vectors.');
end

y = (x(1,:)/4).^2 + (x(2,:)/2).^2 - 1 - sin(2*pi*x(1,:)/4).*(1 - (1/3)*cos(2*pi*x(2,:)/6)) + sin(2*pi*x(2,:)/4);

end