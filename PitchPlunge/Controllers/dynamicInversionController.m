function [out] = dynamicInversionController(h, alpha, hDot, alphaDot, U, t, reward)
%dynamicInversionController This function applies dynamic inversion/partial feedback linearization to control the system. This controller is an adaptation of the controller given by Ko, Kurdila
% and Strganac in their paper "Nonlinear Control of a Prototypical Wing Section with Torsional Nonlinearity". The main problem is that, contrary to the claims made in the paper, the given dynamic
% inversion controller is not internally stable. So this controller often crashes.

% We retrieve parameters from the workspace.
global a b xa m Ia rho ka ca kh ch useNonlinear ka1 ka2 ka3 ka4 cla clb cma cmb;

% We calculate some modified system parameters.
d = m*(Ia - m*xa^2*b^2);
k1 = Ia*kh/d;
k2 = (Ia*rho*b*cla + m*xa*b^3*rho*cma)/d;
k3 = -m*xa*b*kh/d;
k4 = (-m*xa*b^2*rho*cla - m*rho*b^2*cma)/d;
c1 = (Ia*(ch + rho*U*b*cla) + m*xa*rho*U*b^3*cma)/d;
c2 = (Ia*rho*U*b^2*cla*(1/2 - a) - m*xa*b*ca + m*xa*rho*U*b^4*cma*(1/2 - a))/d;
c3 = (-m*xa*b*ch - m*xa*rho*U*b^2*cla - m*rho*U*b^2*cma)/d;
c4 = (m*ca - m*xa*rho*U*b^3*cla*(1/2 - a) - m*rho*U*b^3*cma*(1/2 - a))/d;
g3 = (-Ia*rho*b*clb - m*xa*b^3*rho*cmb)/d;
g4 = (m*xa*b^2*rho*clb + m*rho*b^2*cmb)/d;

% We set up the state.
x1 = h;
x2 = alpha;
x3 = hDot;
x4 = alphaDot;

% We calculate the transformed state phi.
phi1 = x2;
phi2 = x4;
phi3 = x1;
phi4 = -g3*x4 + g4*x3;

% We calculate the current value of ka, as well as p and q.
kaCurrent = ka*(1 + useNonlinear*(ka1*alpha + ka2*alpha^2 + ka3*alpha^3 + ka4*alpha^4));
p = (-m*xa*b/d)*kaCurrent;
q = (m/d)*kaCurrent;

% We now calculate PU and QU.
PU = k4*U^2 + q;
QU = k2*U^2 + p;

% Here's the fun part. We set v. This determines the behavior of the system.
v = -1.2*alphaDot - 4*alpha;

% We now have all the data we need to calculate the control input beta.
beta = (PU*phi1 + (c4 + c3*g3/g4)*phi2 + k3*phi3 + (c3/g4)*phi4 + v)/(U^2*g4);

% I'm replacing the control law from the paper by my own control law. Although, if you look at the definition of phi4, this actually comes down to exactly the same. My law is just a bit easier to
% grasp, once you look into it.
beta = (k3*h+(k4*U^2+q)*alpha + c3*hDot + c4*alphaDot + v)/(U^2*g4);

% Oh, and we should tell the controller that beta is actually the function output.
out = beta;

% We now save some stuff to the logs.
global cLogs cCounter;
cLogs(cCounter,:) = [out,v];

end

