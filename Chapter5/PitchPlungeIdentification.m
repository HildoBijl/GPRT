% This file tests the SONIG algorithm on the pitch-plunge-system with a noisy sinusoud input.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?

% We add paths to folder which contain functions we will use.
addpath('../ExportFig');
addpath('../NIGP/');
addpath('../NIGP/util/');
addpath('../NIGP/tprod/');
addpath('../SONIG/');
addpath('../PitchPlunge/');
addpath('../PitchPlunge/Definitions/');
addpath('../PitchPlunge/Controllers/');
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

%% This block generates measurement data.

% We set up some simulation data.
simdt = 0.1; % This is the time between two successive measurements in the system. (It is not the time step used by Simulink for its own simulation.)
period = 2.5; % This is the period of the sine signal we feed to the system.
amplitude = 0.5; % This is the amplitude of the sine signal.
noiseRange = amplitude/8; % This is the range of the noise added to the sine signal. So the maximum noise is +noiseRange and the minimum noise is -noiseRange.
T = 24*period; % This is the duration of the simulation.
numPoints = T/simdt+1; % This is the number of measurement points we will obtain.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
[x,dx,u,simTime] = createPPSSineInputData(numPoints, simdt, amplitude, 1/period, noiseRange, [0;0], [0;0]);

% We define how many measurements we use for training. (And the rest is for testing.)
nm = numPoints - 4*period/simdt;

% We make some plots of the system data.
figure(1);
clf(1);
hold on;
grid on;
plot(simTime(1:nm), x(1,1:nm), '-', 'Color', red);
plot(simTime(nm:end), x(1,nm:end), '-', 'Color', black);
xlabel('t [s]');
ylabel('h [m]');
if exportFigs ~= 0
	export_fig('PPHeight.png','-transparent');
end

figure(2);
clf(2);
hold on;
grid on;
plot(simTime(1:nm), x(2,1:nm), '-', 'Color', red);
plot(simTime(nm:end), x(2,nm:end), '-', 'Color', black);
xlabel('t [s]');
ylabel('\alpha [rad]');
if exportFigs ~= 0
	export_fig('PPAlpha.png','-transparent');
end

figure(3);
clf(3);
hold on;
grid on;
plot(simTime, u(1,:), '-', 'Color', blue);
xlabel('t [s]');
ylabel('\beta [rad]');
axis([0,10,-0.6,0.600001]);
if exportFigs ~= 0
	export_fig('PPInput.png','-transparent');
end

%% This block will implement all the measurements into the SONIG object.

% We now distort the measurements by noise. Because why would we need an algorithm capable of dealing with noisy input if there is no noisy input?
sx = [0.1;0.6]*1e-3;
xm = x + diag(sx)*randn(size(x));

% % We apply hyperparameter tuning using NIGP to get an initial estimate of the hyperparameters.
% xm = [u(2:nm-1);x(:,1:nm-2);x(:,2:nm-1)];
% fmh = x(:,3:nm);
% [model, nigp] = trainNIGP(xm',fmh',-500,1);
% hyp = NIGPModelToHyperparameters(model);

% We manually define the hyperparameters.
lu = 1;
lx = [0.05;0.3];
su = 0.5*1e-3;
sx = sx; % This is the most useful line ever...
hyp.lx = repmat([lu;lx;lx],[1,2]);
hyp.ly = lx;
hyp.sx = [su;sx;sx];
hyp.sy = sx;

% Next, we set up a SONIG object which we can apply GP regression on.
sonig = createSONIG(hyp);
sonig.addIIPDistance = 0.25; % This is the distance (normalized with respect to the length scales) above which new inducing input points are added.

% We now start to implement measurements.
tic;
lastDisplay = toc;
disp('Starting to implement measurements.');
xpo = xm(:,1:nm); % This will contain the posterior mean of x.
upo = u(:,1:nm); % This will contain the posterior mean of u.
xstd = repmat(sx,[1,size(xpo,2)]); % This will contain the poster standard deviation of x.
ustd = repmat(su,[1,size(upo,2)]); % This will contain the poster standard deviation of u.
jointMean = [0*su;0*sx;xm(:,1);xm(:,2)]; % This will contain the mean vector of the SONIG input which we're currently applying.
jointCov = [su.^2,zeros(1,6);zeros(6,1),blkdiag(diag(sx.^2),diag(sx.^2),diag(sx.^2))]; % This will contain the covariance matrix of the SONIG input which we're currently applying.
for i = 3:nm
	% We display regularly timed updates.
	if toc > lastDisplay + 5
		disp(['Time passed is ',num2str(toc),' seconds. We are currently at measurement ',num2str(i),' of ',num2str(nm),', with ',num2str(sonig.nu),' IIPs.']);
		lastDisplay = toc;
	end
	% We set up the input and the output distributions, taking into account all the covariances of the parameters.
	jointMean = [u(i-1);jointMean(4:7);xm(:,i)]; % We shift the mean matrix one further.
	jointCov = [su.^2,zeros(1,6);zeros(6,1),[jointCov(4:7,4:7),zeros(4,2);zeros(2,4),diag(sx.^2)]]; % We shift the covariance matrix one further.
	jointDist = createDistribution(jointMean, jointCov);
	inputDist = getSubDistribution(jointDist, 1:5);
	outputDist = getSubDistribution(jointDist, 6:7);
	% We implement the measurement into the SONIG algorithm.
	[sonig, inputPost, outputPost, jointPost] = implementMeasurement(sonig, inputDist, outputDist, jointDist);
	% We update the distributions of all our points.
	jointMean = jointPost.mean;
	jointCov = jointPost.cov;
	upo(:,i-1) = jointPost.mean(1);
	xpo(:,i-2:i) = reshape(jointPost.mean(2:7),[2,3]);
	stds = sqrt(diag(jointCov)');
	ustd(:,i-1) = stds(1);
	xstd(:,i-2:i) = reshape(stds(2:7),[2,3]);
end
disp(['Finished implementing ',num2str(sonig.nm),' measurements in ',num2str(toc),' seconds, using ',num2str(sonig.nu),' IIPs.']);

%% This block will evaluate the learned results by making predictions of the future.

% We run a simulation based on the output of the system, for the trial data.
nt = numPoints - nm + 1;
xpo = x(:,nm:end); % This will contain the posterior mean of x.
xstd = repmat(sx,[1,size(xpo,2)]); % This will contain the poster standard deviation of x.
xDist = createDistribution(x(:,nm-1), diag(sx.^2));
newxDist = createDistribution(x(:,nm), diag(sx.^2));
xpo(:,1) = x(:,nm);
for i = nm+1:numPoints
	prevxDist = xDist;
	xDist = newxDist;
	uDist = createDistribution(u(:,i), diag(su.^2));
	inputDist = joinDistributions(joinDistributions(uDist, prevxDist), xDist);
	newxDist = makeSonigStochasticPrediction(sonig, inputDist);
	xpo(:,i-nm+1) = newxDist.mean;
	xstd(:,i-nm+1) = sqrt(diag(newxDist.cov));
end

% We plot the simulation data which we made.
t = (0:simdt:(nt-1)*simdt) + (nm-1)*simdt;
figure(5);
clf(5);
hold on;
grid on;
xlabel('t [s]');
ylabel('h [m]');
patch([t,fliplr(t)], [xpo(1,:)-2*xstd(1,:), fliplr(xpo(1,:)+2*xstd(1,:))], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none');
patch([t,fliplr(t)], [xpo(1,:)-xstd(1,:), fliplr(xpo(1,:)+xstd(1,:))], 1, 'FaceColor', grey, 'EdgeColor', 'none');
set(gca,'layer','top');
plot(t, xpo(1,:), '-', 'Color', blue, 'LineWidth', 1);
plot(t, x(1,nm:end), '-', 'Color', black);
axis([min(t),max(t),-0.05,0.05]);
if exportFigs ~= 0
	export_fig('PPHeightPrediction.png','-transparent');
end

% And also for the other state.
figure(6);
clf(6);
hold on;
grid on;
xlabel('t [s]');
ylabel('\alpha [rad]');
patch([t,fliplr(t)], [xpo(2,:)-2*xstd(2,:), fliplr(xpo(2,:)+2*xstd(2,:))], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none');
patch([t,fliplr(t)], [xpo(2,:)-xstd(2,:), fliplr(xpo(2,:)+xstd(2,:))], 1, 'FaceColor', grey, 'EdgeColor', 'none');
set(gca,'layer','top');
plot(t, xpo(2,:), '-', 'Color', blue, 'LineWidth', 1);
plot(t, x(2,nm:end), '-', 'Color', black);
axis([min(t),max(t),-0.5,0.5]);
if exportFigs ~= 0
	export_fig('PPAlphaPrediction.png','-transparent');
end

% We run another simulation based on the output of the system, but now we manually reduce the covariance of the state at each time. If the algorithm believes it is more certain of its own
% predictions, can it then predict further into the future?
initialStdFactor = 20; % We increase the initial standard deviation (at time zero) by this factor.
stdReduction = 1.05; % We reduce the standard deviation of the predictions by this factor at every time step.
nt = numPoints - nm + 1;
xpo = x(:,nm:end); % This will contain the posterior mean of x.
xstd = repmat(sx,[1,size(xpo,2)]); % This will contain the poster standard deviation of x.
xDist = createDistribution(x(:,nm-1), diag(sx.^2)*initialStdFactor^2);
newxDist = createDistribution(x(:,nm), diag(sx.^2)*initialStdFactor^2);
xpo(:,1) = x(:,nm);
for i = nm+1:numPoints
	newxDist.cov = newxDist.cov/sqrt(stdReduction); % Here we just reduce the covariance after every time step by a given factor. It's a bit of a random thing to do, but somehow it does have very interesting effects.
	prevxDist = xDist;
	xDist = newxDist;
	uDist = createDistribution(u(:,i), diag(su.^2));
	inputDist = joinDistributions(joinDistributions(uDist, prevxDist), xDist);
	newxDist = makeSonigStochasticPrediction(sonig, inputDist);
	xpo(:,i-nm+1) = newxDist.mean;
	xstd(:,i-nm+1) = sqrt(diag(newxDist.cov));
end

% We plot the simulation data which we made.
t = 0:simdt:(nt-1)*simdt;
figure(7);
clf(7);
hold on;
grid on;
xlabel('t [s]');
ylabel('h [m]');
patch([t,fliplr(t)], [xpo(1,:)-2*xstd(1,:), fliplr(xpo(1,:)+2*xstd(1,:))], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none');
patch([t,fliplr(t)], [xpo(1,:)-xstd(1,:), fliplr(xpo(1,:)+xstd(1,:))], 1, 'FaceColor', grey, 'EdgeColor', 'none');
set(gca,'layer','top');
plot(t, xpo(1,:), '-', 'Color', blue, 'LineWidth', 1);
plot(t, x(1,nm:end), '-', 'Color', black);
% axis([min(t),max(t),-0.05,0.05]);
if exportFigs ~= 0
	export_fig('PPHeightPredictionWithReducedCovariance.png','-transparent');
end

% And also for the other state.
figure(8);
clf(8);
hold on;
grid on;
xlabel('t [s]');
ylabel('\alpha [rad]');
patch([t,fliplr(t)], [xpo(2,:)-2*xstd(2,:), fliplr(xpo(2,:)+2*xstd(2,:))], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none');
patch([t,fliplr(t)], [xpo(2,:)-xstd(2,:), fliplr(xpo(2,:)+xstd(2,:))], 1, 'FaceColor', grey, 'EdgeColor', 'none');
set(gca,'layer','top');
plot(t, xpo(2,:), '-', 'Color', blue, 'LineWidth', 1);
plot(t, x(2,nm:end), '-', 'Color', black);
% axis([min(t),max(t),-0.5,0.5]);
if exportFigs ~= 0
	export_fig('PPAlphaPredictionWithReducedCovariance.png','-transparent');
end