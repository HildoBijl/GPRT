% This file contains all the scripts for Chapter 1 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter1 from the Matlab command.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

%% Figure 1.1.
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.001; % This is the output noise scale.
xMin = 0; % This is the minimum value for x.
xMax = 4; % This is the maximum value for x.
nm = 4; % This is the number of measurements we will do.
rng(14, 'twister'); % We fix the random number generator to a state which I know works well, so we always get the same useful outcome.

% We set up input points and measurement data.
Xs = xMin:0.01:xMax; % These are the trial points.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
fmh = sin(xMin + Xm*(2*pi)/(xMax - xMin))' + sfh*randn(size(Xm))'; % These are the measurement values, corrupted by noise.
rng('shuffle'); % And we unfix the random number generator again.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lf^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfh = sfh^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfh)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfh)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(1);
clf(1);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm, fmh, 'ko'); % We plot the measurement points.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm, fmh, 'ro'); % We plot the measurement points.
end
if exportFigs ~= 0
	export_fig('GPRegressionExample.png','-transparent');
end