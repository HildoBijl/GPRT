% In this file we run a basic simulation of the pitch-plunge system. You can use it as template for your own simulations.

% We start off with an empty workspace.
clear all;

% We add paths containing files which we will need.
addpath('Definitions/');
addpath('Controllers/');

% To start off, we define timing data.
T = 10; % We define the simulation length.
dt = 0.01; % We define the simulation time step.
numDataPoints = ceil(T/dt)+1; % We calculate the number of data points we'll be having.

% We also set up other parameters. These are set to default values.
defineFlow; % We set up the flow properties.
defineStructuralParameters; % We define the structural parameters.
defineInitialConditions; % We define the initial conditions.
defineControllerParameters; % We set up the controller parameters.

% We may adjust a couple of the parameters defined above.
% controller = @randomController; % We set a different controller than the default.
h0 = 0; % Initial plunge. [m]
a0 = 0.1; % Initial pitch angle. [rad]
U0 = 15; % Wind speed. [m/s]

% Optionally, this part of the code sets up the optimal LQG controller for the linearized system.
applySettings; % We apply the settings which have been set so far. This also defines system matrices, like M, K, D, E and F.
gamma = 0.5; a = (1/2)*log(gamma); % This is the alpha parameter from the LQG cost function. We have gamma^T = e^(2*a*T).
I = eye(4); Z = zeros(4,4); % These are some helpful matrices.
sysA = [zeros(2,2), eye(2); -M\(K+U0^2*D), -M\(C+U0*E)]; % This is the system A matrix.
sysB = [zeros(2,1); M\(U0^2*F)]; % This is the system B matrix.
X = are(sysA + a*I, sysB/R*sysB', Q); % This is the optimal cost matrix.
sysF = R\sysB'*X; % This is the optimal feedback matix.
controller = @stateController; % We tell the system to use the stateController.
global hGain alphaGain hDotGain alphaDotGain; % We make the gain parameters global, so the stateController can access them.
hGain = sysF(1); alphaGain = sysF(2); hDotGain = sysF(3); alphaDotGain = sysF(4); % We set the controller gains of the simulation state controller.

% Do we use the linear or the nonlinear model?
useNonlinear = 0;

% We run the simulation.
applySettings; % We also apply the settings which have been set, also taking into account any possible adjustments that have been made.
t = sim('PitchPlunge');

% We make some plots.
% In case we want to vary plot numbers, we can use the following parameters.
numPlots = 10;
plotVariation = (1-useNonlinear);
plotIndex = 1:length(t);

figure(1+numPlots*plotVariation);
clf(1+numPlots*plotVariation);
plot(t(plotIndex),x(plotIndex,1));
grid on;
xlabel('Time [s]');
ylabel('Plunge [m]');
title('Plunge h versus time t');

figure(2+numPlots*plotVariation);
clf(2+numPlots*plotVariation);
plot(t(plotIndex),x(plotIndex,2));
grid on;
xlabel('Time [s]');
ylabel('Pitch [rad]');
title('Pitch \alpha versus time t');

figure(3+numPlots*plotVariation);
clf(3+numPlots*plotVariation);
plot(t(plotIndex),U(plotIndex));
grid on;
xlabel('Time [s]');
ylabel('Flow velocity [m/s]');
title('Flow velocity U vs time t');

figure(4+numPlots*plotVariation);
clf(4+numPlots*plotVariation);
plot(t(plotIndex),beta(plotIndex));
grid on;
xlabel('Time [s]');
ylabel('Beta [rad]');
title('Input beta vs time t');

figure(5+numPlots*plotVariation);
clf(5+numPlots*plotVariation);
plot(t(plotIndex),xd(plotIndex,1));
grid on;
xlabel('Time [s]');
ylabel('Plunge rate [m/s]');
title('Plunge rate h_{dot} versus time t');

figure(6+numPlots*plotVariation);
clf(6+numPlots*plotVariation);
plot(t(plotIndex),xd(plotIndex,2));
grid on;
xlabel('Time [s]');
ylabel('Pitch rate [rad/s]');
title('Pitch rate alpha_{dot} vs time t');

value = (accReward - accReward(end))./(accWeight - accWeight(end)); % We calculate the value based on the mean (weighted) reward that we will still acquire in the future.
cutOff = 500; % At the end the value estimate isn't accurate anymore. So we simply cut off the end. How many data points do we cut off?
figure(7);
clf(7);
hold on;
grid on;
plot(t(1:end-cutOff),value(1:end-cutOff));
xlabel('Time [s]');
ylabel('State value');
title('State value versus time for the given controller');

figure(8);
clf(8);
hold on;
grid on;
plot3(x(:,1),x(:,2),xd(:,1));
xlabel('Plunge [m]');
ylabel('Pitch [rad]');
zlabel('Plunge rate [m/s]');
title('State procession during the experiments');
view([120,30]);

figure(9);
clf(9);
hold on;
grid on;
plot(t,dAlpha,'b-');
xlabel('Time [s]');
ylabel('d\alpha [rad]');
title('Value of d\alpha due to turbulence over time');

figure(10);
clf(10);
plot(t(plotIndex),reward(plotIndex));
grid on;
xlabel('Time [s]');
ylabel('Reward [-]');
title('Reward given over time (unweighted)');

disp(['The accumulated reward is ',num2str(accReward(end)),'. Analytically, for the linear system, it should be ',num2str(log(gamma)*[x0,xd0]*X*[x0';xd0']),'.']); % These are slightly different. I guess these differences are caused because of the zero-order hold of the input signal.