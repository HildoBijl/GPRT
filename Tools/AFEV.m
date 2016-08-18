function [y] = AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)
%AFEV The acquisition function using the expected value.

	% We make a GP prediction for the test point.
	[mu,~] = getGPPrediction(x, Xu, muu, Suu, Kuu, lf, lx, eps);
	
	% We set up the right output value.
	y = mu;
end

