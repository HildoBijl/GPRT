% In this file we do a first Gaussian process experiment, trying to approximate a part of the dynamics of a simple pitch-plunge system.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?

% We add paths containing files which we will need.
addpath('../PitchPlunge/Definitions/');
addpath('../PitchPlunge/Controllers/');
addpath('../ExportFig/');
addpath('../Tools/');

% We define colors.
black = [0 0 0];
white = [1 1 1];
if useColor == 0
	red = [0 0 0];
	green = [0.6 0.6 0.6];
	blue = [0.2 0.2 0.2];
	yellow = [0.4 0.4 0.4];
	grey = [0.8 0.8 0.8];
else
	red = [0.8 0 0];
	green = [0 0.4 0];
	blue = [0 0 0.8];
	yellow = [0.6 0.6 0];
	grey = [0.8 0.8 1];
end

% Before we start, we first run a general simulation of the system.
T = 4; % We define the simulation length.
dt = 0.01; % We define the simulation time step.
numDataPoints = ceil(T/dt)+1; % We calculate the number of data points we'll be having.

% We also set up other parameters. These are set to default values.
defineFlow; % We set up the flow properties.
defineStructuralParameters; % We define the structural parameters.
defineInitialConditions; % We define the initial conditions.
defineControllerParameters; % We set up the controller parameters.

% We adjust the initial conditions somewhat, making sure flutter takes place.
U0 = 15; % We adjust the wind speed from its default value.
h0 = 0; % Initial plunge. [m]
a0 = 0.1; % Initial pitch angle. [rad]

% We run the simulation.
applySettings; % We also apply the settings which have been set, also taking into account any possible adjustments that have been made.
t = sim('../PitchPlunge/PitchPlunge');

% We make plots of the simulation results.
figure(1);
clf(1);
if useColor == 0
	plot(t,x(:,1),'k-');
else
	plot(t,x(:,1),'b-');
end
grid on;
xlabel('Time [s]');
ylabel('Plunge [m]');
if exportFigs ~= 0
	export_fig('PitchPlungeResponse1.png','-transparent');
end

figure(2);
clf(2);
if useColor == 0
	plot(t,x(:,2),'k-');
else
	plot(t,x(:,2),'b-');
end
grid on;
xlabel('Time [s]');
ylabel('Pitch [rad]');
if exportFigs ~= 0
	export_fig('PitchPlungeResponse2.png','-transparent');
end

% Next, it's time to gather data for GP regression. We set the number of measurements that we want to do.
nm = 30; % We set the number of time steps we want to feed to the GP.
Xm = zeros(2,nm); % This set will contain all input data.
fmh = zeros(nm,2); % This set will contain all output data.
rng(6, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.

% To start off, we define timing data.
dt = 0.001; % We define the simulation time step.
T = 0.1; % We define the simulation length.
numDataPoints = ceil(T/dt)+1; % We calculate the number of data points we'll be having.

% We also set up other parameters. These are set to default values.
defineFlow; % We set up the flow properties.
defineStructuralParameters; % We define the structural parameters.
defineInitialConditions; % We define the initial conditions.
defineControllerParameters; % We set up the controller parameters.

% We define the range for the initial state.
U0 = 15; % We adjust the wind speed from its default value.
h0Range = 5e-3;
a0Range = 6e-2;
hd0Range = 5e-2;
ad0Range = 1e0;
betaRange = 5e-1;
useNonlinear = 1; % We indicate we use the linear model for now.

% We define which controller we will use.
controller = @constantController;
global constantControllerValue;
constantControllerValue = 0;

% We now loop through the experiments to obtain data.
for experiment = 1:nm
	% We set the initial state for the simulation.
	h0 = (rand(1,1)*2-1)*h0Range; % Initial plunge. [m]
	a0 = (rand(1,1)*2-1)*a0Range; % Initial pitch angle. [rad]

	% We run the simulation.
	applySettings; % We also apply the settings which have been set, also taking into account any possible adjustments that have been made.
	t = sim('../PitchPlunge/PitchPlunge');
	pitchPlungeK = K; % We also store the value of the spring stiffness matrix, because soon our GP script will override K with the covariance matrix.

	% We extract data.
	Xm(:,experiment) = x(1,:)';
	fmh(experiment,:) = x(end,:);
end

% For the hyperparameters, we can use the tuning methods of the next chapter.
% [sfm1, lf1, lx1, mb1] = tuneHyperparameters(Xm, fmh(:,1));
% [sfm2, lf2, lx2, mb2] = tuneHyperparameters(Xm, fmh(:,2));
% sfm = [sfm1;sfm2];
% lf = [lf1;lf2];
% lx = [lx1,lx2];
% mb = [mb1;mb2];

% Or we just set the hyperparameters ourselves to sensible values.
xScale = [1e-2;2e-1;1e-1;5e0;5e-1];
sfm = xScale(1:4)'/100;
lf = xScale(1:4)*4;
lx = [xScale,xScale];
mb = zeros(4,1);

% Next, we will plot the next state based on the current and previous state. For this, we first set up the trial input points.
hMin = -h0Range;
hMax = h0Range;
aMin = -a0Range;
aMax = a0Range;
nsPerDimension = 21; % This is the number of trial points per dimension.
ns = nsPerDimension^2; % This is the total number of trial points.
[x1Mesh,x2Mesh] = meshgrid(linspace(hMin,hMax,nsPerDimension),linspace(aMin,aMax,nsPerDimension));
Xs = [reshape(x1Mesh,1,ns); reshape(x2Mesh,1,ns)];
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We calculate the next state, based on the system equations. This is an approximation, because nonlinear effects are not taken into account.
At = [zeros(2,2), eye(2); -M\(pitchPlungeK+U0^2*D), -M\(C+U0*E)];
Bt = [zeros(2,1); M\(U0^2*F)];
Ae = expm(At*T);
newStateLinear = Ae(1:2,1:2)*Xs(1:2,:);

% We apply GP regression for the first output.
% We start making predictions for the plots. We do this for different outputs.
mPostStorage = zeros(nsPerDimension,nsPerDimension,2); % We set up a storage for the posterior mean.
for outputIndex = 1:2
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(1:2,outputIndex).^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
	Kmm = K(1:nm,1:nm);
	Kms = K(1:nm,nm+1:end);
	Ksm = Kms';
	Kss = K(nm+1:end,nm+1:end);
	Sfm = sfm(outputIndex)^2*eye(nm); % This is the noise covariance matrix.
	mm = mb(outputIndex)*ones(nm,1); % This is the mean vector m(Xm). We assume a constant mean function.
	ms = mb(outputIndex)*ones(ns,1); % This is the mean vector m(Xs). We assume a constant mean function.
	mPost = ms + Ksm/(Kmm + Sfm)*(fmh(:,outputIndex) - mm); % This is the posterior mean vector.
	SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	logp = -nm/2*log(2*pi) - 1/2*logdet(Kmm + Sfm) - 1/2*(fmh(:,outputIndex) - mm)'/(Kmm + Sfm)*(fmh(:,outputIndex) - mm); % In case we are interested, we can also calculate the log(p) value.

	% And then we plot the result.
	figure(2+outputIndex);
	clf(2+outputIndex);
	hold on;
	grid on;
	sDown = surface(x1Mesh, x2Mesh, mPost - 2*sPost);
	set(sDown,'FaceAlpha',0.3);
	set(sDown,'LineStyle','none');
	set(sDown,'FaceColor',blue);
	sUp = surface(x1Mesh, x2Mesh, mPost + 2*sPost);
	set(sUp,'FaceAlpha',0.3);
	set(sUp,'LineStyle','none');
	set(sUp,'FaceColor',blue);
	sMid = surface(x1Mesh, x2Mesh, mPost);
	set(sMid,'FaceAlpha',0.8);
	set(sMid,'FaceColor',blue);
	if useColor == 0
		scatter3(Xm(1,:), Xm(2,:), fmh(:,outputIndex), 'ko', 'filled');
	else
		scatter3(Xm(1,:), Xm(2,:), fmh(:,outputIndex), 'ro', 'filled');
	end
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([-110,30])
	end
	if outputIndex == 2
		view([-110,30])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1e-1,1e-1]);
	end
	if exportFigs ~= 0
		export_fig(['NextStatePrediction',num2str(outputIndex),'.png'],'-transparent');
	end

	% We also plot the analytic result.
	sLinear = surface(x1Mesh, x2Mesh, reshape(newStateLinear(outputIndex,:), nsPerDimension, nsPerDimension));
	set(sLinear,'FaceAlpha',0.5);
	set(sLinear,'FaceColor',green);
	
	% And we plot the posterior mean, because we want to save it.
	mPostStorage(:,:,outputIndex) = mPost;
end

% Finally we save some data, so later chapters may use it.
save('CH2Predictions','x1Mesh','x2Mesh','mPostStorage');

% The next step is to apply GP regression for any random initial state, and not just stationary initial states. We also add random inputs.
% We loop through the experiments to obtain data.
Xm = zeros(5,nm); % This is the new size of the Xm matrix.
fmh = zeros(nm,4); % This is the new size of the fmh matrix.
rng(1, 'twister'); % We fix Matlab's random number generator, so we ensure we get the same results all the time, also compared to other experiments in other files.
for experiment = 1:nm
	% We set the initial state for the simulation.
	h0 = (rand(1,1)*2-1)*h0Range; % Initial plunge. [m]
	a0 = (rand(1,1)*2-1)*a0Range; % Initial pitch angle. [rad]
	hd0 = (rand(1,1)*2-1)*hd0Range; % Initial plunge rate. [m/s]
	ad0 = (rand(1,1)*2-1)*ad0Range; % Initial pitch angle rate. [rad/s]
	constantControllerValue = (rand(1,1)*2-1)*betaRange; % Control input. [rad]

	% We run the simulation.
	applySettings; % We apply the settings which have been set so far, also taking into account any possible adjustments that have been made. This also sets up the initial conditions.
	t = sim('../PitchPlunge/PitchPlunge');
	Xm(:,experiment) = [x(1,:)';xd(1,:)';constantControllerValue];
	fmh(experiment,:) = [x(end,:),xd(end,:)];
end

% We set up the difference matrix, preparing ourselves for GP regression.
Xs = [Xs;zeros(3,ns)];
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We apply GP regression for each individual output.
for outputIndex = 1:2
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
	Kmm = K(1:nm,1:nm);
	Kms = K(1:nm,nm+1:end);
	Ksm = Kms';
	Kss = K(nm+1:end,nm+1:end);
	Sfm = sfm(outputIndex)^2*eye(nm); % This is the noise covariance matrix.
	mm = mb(outputIndex)*ones(nm,1); % This is the mean vector m(Xm). We assume a constant mean function.
	ms = mb(outputIndex)*ones(ns,1); % This is the mean vector m(Xs). We assume a constant mean function.
	mPost = ms + Ksm/(Kmm + Sfm)*(fmh(:,outputIndex) - mm); % This is the posterior mean vector.
	SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	logp = -nm/2*log(2*pi) - 1/2*logdet(Kmm + Sfm) - 1/2*(fmh(:,outputIndex) - mm)'/(Kmm + Sfm)*(fmh(:,outputIndex) - mm); % In case we are interested, we can also calculate the log(p) value.

	% And then we plot the result.
	figure(4+outputIndex);
	clf(4+outputIndex);
	hold on;
	grid on;
	sDown = surface(x1Mesh, x2Mesh, mPost - 2*sPost);
	set(sDown,'FaceAlpha',0.3);
	set(sDown,'LineStyle','none');
	set(sDown,'FaceColor',blue);
	sUp = surface(x1Mesh, x2Mesh, mPost + 2*sPost);
	set(sUp,'FaceAlpha',0.3);
	set(sUp,'LineStyle','none');
	set(sUp,'FaceColor',blue);
	sMid = surface(x1Mesh, x2Mesh, mPost);
	set(sMid,'FaceAlpha',0.8);
	set(sMid,'FaceColor',blue);
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1.5e-2,1e-2]);
	end
	if outputIndex == 2
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-2e-1,2e-1]);
	end

	% We also plot the result from the previous chapter.
	load('../Chapter2/CH2Predictions');
	sPrevious = surface(x1Mesh, x2Mesh, mPostStorage(:,:,outputIndex));
	set(sPrevious,'FaceAlpha',0.5);
	set(sPrevious,'FaceColor',green);
	if exportFigs ~= 0
		export_fig(['NextStatePredictionFiveDimensional',num2str(outputIndex),'.png'],'-transparent');
	end
end