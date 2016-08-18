function [out] = randomController(h, alpha, hDot, alphaDot, U, t, reward)
%randomController This controller sets a new random controller input after every fixed set of time.

% We retrieve global parameters.
global cPeriod cLastSwitch cLastValue cRandomMinValue cRandomMaxValue;

% We start by checking if we should set a new controller value. If so, we make it happen.
if t >= cLastSwitch + cPeriod - 1e-9 % We subtract a small number to make sure that numerical fluctuations in the time value do not cause any significant problems.
	cLastValue = cRandomMinValue + rand(1,1)*(cRandomMaxValue - cRandomMinValue);
	cLastSwitch = cLastSwitch + cPeriod;
end

% We set the controller output value to the last controller value that was set.
out = cLastValue;

end