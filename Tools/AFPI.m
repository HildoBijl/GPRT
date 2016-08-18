function [y] = AFPI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xi)
%AFPI The acquisition function using the probability of improvement.
	% We make a GP prediction for the test point.
	[mu,vari] = getGPPrediction(x, Xu, muu, Suu, Kuu, lf, lx, eps);
	
	% We calculate the probability of improvement. We do this using the error function.
	z = ((mu - fOpt - xi)/sqrt(vari));
	Phi = 1/2 + 1/2*erf(z/sqrt(2));
	
	% We set up the right output value.
	res = Phi;
	if res > 0
		y = log(Phi); % Usually the output is Phi, but we take the logarithm because otherwise the values are just too small.
	else
		y = -1e200; % Sometimes y becomes zero for numerical reasons. This basically means that it's extremely small. Still, it'll crash the algorithm. So we just set a default very small value here.
	end
end

