% In this file the parameters for the controller are defined. Which
% controller do we use and how does it work?

% First we define which actual controller we will be using.
global controller;
controller = @constantController;
% controller = @dynamicInversionController;
% controller = @stateController;

% We set up the constant controller.
global constantControllerValue;
constantControllerValue = 0;

% We set up the random controller.
global cPeriod cRandomMinValue cRandomMaxValue;
cPeriod = 0.1;
cRandomMinValue = -0.5;
cRandomMaxValue = 0.5;

% For the state controller, we define reference signals and gains.
global hRef alphaRef hDotRef alphaDotRef hGain alphaGain hDotGain alphaDotGain;
hRef = 0;
alphaRef = 0;
hDotRef = 0;
alphaDotRef = 0;
hGain = 10;
alphaGain = -1/4;
hDotGain = 0;
alphaDotGain = 0;

% We define the value of gamma, which the controller uses to calculate the accumulated reward, which is used as a quality measure of the controller.
global gamma;
gamma = 0.5;

% We define the damping (the gamma) of the state filter. This is similar to the value for gamma. It is related to the filter time constant through tau = -1/log(stateFilterGamma).
stateFilterGamma = gamma;

% We also define the other parameters for the reward function.
hBar = 5e-3; % What's the normalizing constant for h in the reward function?
alphaBar = 6e-2; % What's the normalizing constant for alpha in the reward function?
hDotBar = 5e-2; % What's the normalizing constant for hDot in the reward function?
alphaDotBar = 1e0; % What's the normalizing constant for alphaDot in the reward function?
betaBar = 5e-1; % What's the normalizing constant for beta in the reward function?
crh = 0; % What's the reward function coefficient for h?
cralpha = 0; % What's the reward function coefficient for alpha?
crhDot = 2; % What's the reward function coefficient for hDot?
cralphaDot = 2; % What's the reward function coefficient for alphaDot?
crbeta = 4; % What's the reward function coefficient for beta?
Q = zeros(4,4); Q(1,1) = crh/hBar^2; Q(2,2) = cralpha/alphaBar^2; Q(3,3) = crhDot/hDotBar^2; Q(4,4) = cralphaDot/alphaDotBar^2; % We define the Q-matrix of the cost function.
R = crbeta/betaBar^2; % We define the R matrix of the cost function.

% We initialize storage parameters. Oh, by the way, the preliminary "c" is short for "controller". cIn stores the input, while cOut stores the output. The cLogs parameter can optionally be used
% by the controller to log data at each time step.
global cCounter cIn cOut cLogs;
cCounter = 0;
cIn = zeros(ceil(T/dt)+1,7); % This will keep track of data fed into the controller.
cOut = zeros(ceil(T/dt)+1,1); % This will keep track of data given by the controller.