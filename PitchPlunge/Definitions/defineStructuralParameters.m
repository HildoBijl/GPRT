% This file defines several structural parameters of the pitch-plunge system.

% We want all parameters here to be global, so they can be accessed by Simulink at any point during the simulation.
global a b c xa m Ia rho ka ca kh ch useNonlinear ka1 ka2 ka3 ka4 cla clb cma cmb;

% Wing dimensions.
c = 0.270; % Wing chord. [m]
b = c/2; % Wing semichord. [m]
a = -0.6; % Nondimensionalized distance from the midchord to the rotation point. [m]
xa = 0.2466; % Nondimensionalized distance from the rotation point to the CG. [-]

% Inertia properties.
m = 12.387; % Wing mass. [kg]
Ia = 0.065; % Wing inertia. [kgm^2]
rho = 1.225; % Air density. [kg/m^3]

% Wing linear structural coefficients.
ka = 2.82; % Pitch structural stiffness coefficient. [Nm/rad] Note: normally this parameter isn't constant but actually depends nonlinearly on alpha. We approximate it with a constant.
ca = 0.036; % Pitch structural damping coefficient. [Ns]
kh = 2844.4; % Plunge structural stiffness coefficient. [Nm/m]
ch = 27.43; % Plunge structural damping coefficient. [kg/s]

% Wing nonlinear structural coefficients.
useNonlinear = 1; % Boolean setting: 0 for linear or 1 for nonlinear.
ka1 = -22.1;
ka2 = 1315.5;
ka3 = -8580;
ka4 = 17289.7;

% Aerodynamic coefficients.
cla = 6.28; % Lift coefficient derivative. [-]
clb = 3.358; % Control surface lift effectiveness. [-]
cma = (0.5+a)*cla; % Moment coefficient derivative. [-]
cmb = -0.635; % Control surface moment effectiveness. [-]