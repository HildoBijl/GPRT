% This file contains a constant controller, which returns the value set in the global constantControllerValue parameter.
function out = constantController(h, alpha, hDot, alphaDot, U, t, reward)
	global constantControllerValue;
	out = constantControllerValue;
end