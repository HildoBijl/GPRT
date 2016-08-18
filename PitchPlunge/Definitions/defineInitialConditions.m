% This file defines initial conditions for the Pitch-Plunge system.

% These are the zero initial conditions.
h0 = 0; % Initial plunge. [m]
a0 = 0; % Initial pitch angle. [rad]
hd0 = 0; % Initial plunge velocity. [m/s]
ad0 = 0; % Initial pitch rate. [rad/s]

% This is the original set of initial conditions from the "Nonlinear Control of a Prototypical Wing Section with Torsional Nonlinearity" paper. When applying these initial conditions, together
% with U0 = 15 m/s and zero control, we (should) get the plots given by that paper. (See Figures 2 and 3 from the paper.)
% h0 = 0.01; % Initial plunge. [m]
% a0 = 0.1; % Initial pitch angle. [rad]
% hd0 = 0; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% This is a set of initial conditions causing some flutter if there is no controller. It's one that I often used as an initial point.
% h0 = -0.0026; % Initial plunge. [m]
% a0 = 0.1; % Initial pitch angle. [rad]
% hd0 = -0.008; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% This set causes oscillations with a zero controller. The fluttering is not damped out. (When velocity is constant at 10 m/s.)
% h0 = 0.002; % Initial plunge. [m]
% a0 = 0.0035; % Initial pitch angle. [rad]
% hd0 = 0; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% Surprisingly, this set is stable with a zero controller. The fluttering is damped out. (When velocity is constant at 10 m/s.)
% h0 = 0.002; % Initial plunge. [m]
% a0 = 0.0033; % Initial pitch angle. [rad]
% hd0 = 0; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% This is a semi-random test initial state, which flutters.
% h0 = 0.0026; % Initial plunge. [m]
% a0 = 0.1; % Initial pitch angle. [rad]
% hd0 = -0.008; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% This is a semi-random test initial state, which stabilizes.
% h0 = 0.001; % Initial plunge. [m]
% a0 = 0.001; % Initial pitch angle. [rad]
% hd0 = 0; % Initial plunge velocity. [m/s]
% ad0 = 0; % Initial pitch rate. [rad/s]

% These are zero initial conditions. Or at least, close to zero, to prevent absolute zeroes which can be complicated when dividing by the error.
% h0 = 1e-9; % Initial plunge. [m]
% a0 = 1e-9; % Initial pitch angle. [rad]
% hd0 = 1e-9; % Initial plunge velocity. [m/s]
% ad0 = 1e-9; % Initial pitch rate. [rad/s]