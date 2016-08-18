function [y] = AFUCB(x, Xu, muu, Suu, Kuu, lf, lx, eps, kappa)
%AFUCB The acquisition function using the upper confidence bound.

	% We make a GP prediction for the test point.
	[mu,vari] = getGPPrediction(x, Xu, muu, Suu, Kuu, lf, lx, eps);
	
	% We set up the right output value.
	y = mu + kappa*sqrt(vari);
end

