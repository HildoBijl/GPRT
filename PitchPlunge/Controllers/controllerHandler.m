function [out] = controllerHandler(in)
%controller This function handles the controller routing of the Pitch Plunge model. It checks which controller needs to be executed and executes it. (This controller is defined in the global
%'controller' parameter.) Furthermore, this function also stores input and output data.

% Often Matlab first calls this function with zero inputs. In this case we do not have to store any data or do any fancy stuff.
if in == zeros(size(in))
	out = zeros(1,1);
 	return;
end

% We start off by storing the given data.
global cCounter cIn cOut;
cCounter = cCounter + 1;
cIn(cCounter,:) = in;

% We extract parameters from the in-array.
t = in(1);
h = in(2);
alpha = in(3);
hDot = in(4);
alphaDot = in(5);
U = in(6);
reward = in(7);

% We check which controller we should use and apply that controller. That is where the magic happens.
global controller;
out = controller(h, alpha, hDot, alphaDot, U, t, reward);

% Finally, we apply data storage for the output too.
cOut(cCounter,:) = out;

end

