function [state, dstate, input, time] = createPPSSineInputData(n, resdt, amplitude, frequency, noiseRange, x0, xd0)
%createPPSSineInputData Creates data points for the pitch-plunge-system subject to a sine input with a given frequency.
% This function creates a set of data points which can be used to identify the pitch-plunge system.
%
% Inputs are:
%	n - the number of data points that we want. (Default 100.)
%	resdt - the time step that we use in between data points. (This is generally not the simulation time step. Default 0.1.)
%	amplitude - the amplitude of the sine input given to the system. (Default 1.)
%	frequency - the frequency of the sine input given to the system. (Default 1.)
%	noiseRange - the maximum of the extra noise that we apply to the (sine) input signal, according to a uniform distribution. (Default 0.)
%	x0 - the initial state [h;alpha] of the system. (Default a zero vector.)
%	dx0 - the initial state derivative of the system. (Default a zero vector.)
%
% Outputs of the function are the input and output of the output prediction function.
%	x - this is the state [h;alpha] of the system at each point in time.
%	dx - this is the state derivative of the system at each point in time.
%	u - this is the input given to the system at each point in time.

% We check the parameters we received and give default values wherever necessary.
if nargin < 7
	xd0 = [0;0];
end
if nargin < 6
	x0 = [0;0];
end
if nargin < 5
	noiseRange = 0;
end
if nargin < 4
	frequency = 1;
end
if nargin < 3
	amplitude = 1;
end
if nargin < 2
	resdt = 0.1;
end
if nargin < 1
	n = 100;
end

% We calculate timing data. For this, we need to make sure that the time parameters are adjusted in the base workspace, which Simulink uses. So we set both our current workspace and the base
% workspace for these parameters equal to the global workspace.
evalin('base','clear dt T numDataPoints; global dt T numDataPoints;');
global dt T numDataPoints;
dt = 0.01; % This is the maximum timestep dt applied in the numerical simulation. We'll slightly adjust it so it fits an integer number of times in the given value of resdt.
numTimeStepsPerInterval = ceil(resdt/dt); % This is the number of time steps applied in the numerical simulation per time interval dt. (For instance, if resdt = 0.1 and dt = 0.01, then we take ten time steps in our numerical simulation before we return another data point.)
dt = resdt/numTimeStepsPerInterval; % This is the timestep dt applied in the numerical simulation.
T = (n-1)*numTimeStepsPerInterval*dt; % This is the full simulation time.
numDataPoints = (n-1)*numTimeStepsPerInterval + 1; % We tell the main workspace (from which all simulation data is drawn) what the number of data points we will use is.

% We set up the conditions for the simulation to run in. We do this from the base workspace, since this is where Simulink gets its data from.
evalin('base','defineFlow');
evalin('base','defineStructuralParameters');
evalin('base','defineInitialConditions');
evalin('base','defineControllerParameters');

% We set up the initial conditions.
evalin('base','clear h0 a0 hd0 ad0; global h0 a0 hd0 ad0;');
global h0 a0 hd0 ad0;
h0 = x0(1);
a0 = x0(2);
hd0 = xd0(1);
ad0 = xd0(2);

% We apply our settings (at least the initial condition).
evalin('base','applySettings');

% We also set up a controller. For this, we define various necessary parameters.
global controller;
controller = @noisySineInput;
global sineAmplitude sineFrequency;
sineAmplitude = amplitude;
sineFrequency = frequency;
global cPeriod cLastSwitch cLastValue cRandomMinValue cRandomMaxValue;
cPeriod = resdt;
cLastSwitch = -cPeriod;
cLastValue = 0;
cRandomMinValue = -noiseRange;
cRandomMaxValue = noiseRange;

% We make the parameters that we want to extract global.
evalin('base','clear x beta; clear xd beta; global x beta;');
global x xd beta;

% We run the simulation.
sim('PitchPlunge');

% We extract the points which we actually use and right away corrupt them with noise.
state = x(1:numTimeStepsPerInterval:end,:)';
dstate = xd(1:numTimeStepsPerInterval:end,:)';
input = beta(1:numTimeStepsPerInterval:end,:)';
time = 0:resdt:T;

end