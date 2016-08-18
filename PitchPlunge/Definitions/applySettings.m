% In this file we do some final calculations based on the settings that have been provided. It makes everything ready for the simulation to run.

% We merge the initial conditions into initial vectors.
x0 = [h0,a0];
xd0 = [hd0,ad0];

% We set up system matrices.
M = [m,m*xa*b;m*xa*b,Ia]; % Inertia matrix.
Minv = inv(M); % Inertia matrix inverse.
C = [ch,0;0,ca]; % Damping matrix.
K = [kh,0;0,ka]; % Stiffness matrix.
D = [0,rho*b*cla;0,-rho*b^2*cma]; % Static aerodynamic force matrix.
E = [rho*b*cla,rho*b^2*cla*(1/2-a);-rho*b^2*cma,-rho*b^3*cma*(1/2-a)]; % Dynamic aerodynamic force matrix.
F = [-rho*b*clb;rho*b^2*cmb]; % Input effectiveness matrix.

% Here we arrange the flow disturbance. We calculate the magnitude of the disturbance at each point in time, and then set it up in a Simulink-suitable format.
disturbanceDeviation = (disturbanceTime-disturbanceCenter).^2; % For every point, we calculate the squared distance from the disturbance center.
disturbance = disturbanceMagnitude * 0.5.*(2/(sqrt(3*disturbanceRadius)*pi^0.25)) * exp(-disturbanceDeviation/(2*disturbanceRadius^2)) .* (1-disturbanceDeviation/(disturbanceRadius^2)); % We calculate a Mexican-hat-shaped disturbance.
dU = [disturbanceTime',disturbance']; % We put it in a Simulink-suitable format.

% We finalize some controller settings.
global cLastSwitch;
cLastSwitch = -cPeriod;
