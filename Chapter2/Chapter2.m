% This file contains all the scripts for Chapter 2 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter2 from the Matlab command.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

%% Figure 2.1.
disp('Creating Figure 2.1.');
% We define data.
mPrior = 0; % This is the prior mean.
lfPrior = 2; % This is the prior standard deviation.
fHat = 2; % This is the measured value.
sfHat = 1.5; % This is the measurement noise standard deviation.

% We merge the data together to find the posterior mean and PDF.
sPost = sqrt(inv(inv(lfPrior^2) + inv(sfHat^2))); % This is the posterior standard deviation.
mPost = sPost^2*(lfPrior^2\mPrior + sfHat^2\fHat); % This is the posterior mean.

% We now set up plot data.
x = -5:0.01:5;
pdfPrior = normpdf(x, mPrior, lfPrior);
pdfMeasurement = normpdf(x, fHat, sfHat);
pdfProduct = pdfPrior.*pdfMeasurement; % We multiply the two PDFs.
pdfPosterior = normpdf(x, mPost, sPost); % And we normalize the result by dividing by the integral value. (Which is equivalent to calculating the PDF with the right mean and standard deviation, which is what we actually do.)

% We make a plot.
figure(1);
clf(1);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	plot(x, pdfPrior, 'k-');
	plot(x, pdfMeasurement, 'k--');
	plot(x, pdfProduct, 'k-.');
	plot(x, pdfPosterior, 'k-','LineWidth',2);
else
	plot(x, pdfPrior, 'k-');
	plot(x, pdfMeasurement, 'r-');
	plot(x, pdfProduct, 'g-');
	plot(x, pdfPosterior, 'b-');
end
legend('Prior distribution','Measurement distribution','Multiplication','Posterior distribution','Location','NorthWest');

% We export the plot, if desired.
if exportFigs ~= 0
	export_fig('MergingDistributions.png','-transparent');
end

%% Figure 2.2.
disp('Creating Figure 2.2.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sf = 0.7; % This is the output noise scale.
X = [1,2,3]; % This is the set of input points x(1), x(2) and x(3).
fh = cos(X - 2)'; % This is the measured function value. The function we will approximate is cos(x - 2).

% We now set up the (squared exponential) covariance matrix and some related parameters.
n = size(X,2); % This is the number of input points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
m = zeros(n,1); % This is the mean vector m(X). We assume a zero mean function.
Sfh = sf^2*eye(n); % This is the noise matrix \hat{S}_f.

% We apply GP regression.
SPost = inv(inv(K) + inv(Sfh)); % This is the posterior covariance matrix.
mPost = SPost*(K\m + Sfh\fh); % This is the posterior mean vector.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the first plot with the prior error bars.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, m, 2*lf*ones(n,1), 2*lf*ones(n,1), 'kd-');
else
	errorbar(X, m, 2*lf*ones(n,1), 2*lf*ones(n,1), 'ko-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('MergingMultipleDistributionsPrior.png','-transparent');
end

% We set up the second plot, with the measured error bars.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, fh, 2*sf*ones(n,1), 2*sf*ones(n,1), 'ko-');
else
	errorbar(X, fh, 2*sf*ones(n,1), 2*sf*ones(n,1), 'ro-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('MergingMultipleDistributionsMeasurement.png','-transparent');
end

% We set up the third plot, with the posterior error bars.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, mPost, 2*sPost, 2*sPost, 'k*-');
else
	errorbar(X, mPost, 2*sPost, 2*sPost, 'bo-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('MergingMultipleDistributionsPosterior.png','-transparent');
end

%% Figure 2.3.
disp('Creating Figure 2.3.');
% We define and calculate data.
lx = 1;
x = -3:0.01:3;
c = exp(-1/2*x.^2/lx^2);

% We plot the figure.
figHandle = figure(3);
set(figHandle, 'Position', [500, 500, 800, 300]); % We want a different size for this plot, so we adjust that.
clf(3);
hold on;
grid on;
plot(x,c,'k-');
xlabel('Input');
ylabel('Correlation c(x,x'')');
if exportFigs ~= 0
	export_fig('CorrelationFunction.png','-transparent');
end

%% Figure 2.4.
disp('Creating Figure 2.4.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
X = [1,2,3]; % This is the set of input points x(1), x(2) and x(3).
f1 = cos(X(1) - 2); % This is the measured function value. The function we will approximate is cos(x - 2).

% We now set up the (squared exponential) covariance matrix.
n = size(X,2); % This is the number of input points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
m = zeros(n,1); % This is the mean vector m(X). We assume a zero mean function.

% We apply GP regression.
mPost = m + K(:,1)/K(1,1)*(f1 - m(1)); % This is the posterior mean vector.
SPost = K - K(:,1)/K(1,1)*K(1,:); % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the first plot with the prior error bars.
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, m, 2*lf*ones(n,1), 2*lf*ones(n,1), 'kd-');
else
	errorbar(X, m, 2*lf*ones(n,1), 2*lf*ones(n,1), 'ko-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('FirstGPRegressionPrior.png','-transparent');
end

% We set up the second plot, with the measured error bars.
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, [f1;0;0], [0;1e6;1e6], [0;1e6;1e6], 'ko-'); % We use 1e6 to indicate a very big (infinite) number.
else
	errorbar(X, [f1;0;0], [0;1e6;1e6], [0;1e6;1e6], 'ro-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('FirstGPRegressionMeasurement.png','-transparent');
end

% We set up the third plot, with the posterior error bars.
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(X, mPost, 2*sPost, 2*sPost, 'k*-');
else
	errorbar(X, mPost, 2*sPost, 2*sPost, 'bo-');
end
axis([0.5,3.5,-2.5,2.5]);
if exportFigs ~= 0
	export_fig('FirstGPRegressionPosterior.png','-transparent');
end

%% Figure 2.5.
disp('Creating Figure 2.5.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
Xs = 0.2:0.2:3.8; % These are the trial points.
Xm = [1,2.5]; % These are the measurement points.
fmh = cos(Xm - 2)'; % These are the measurement values.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/Kmm*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/Kmm*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up error bar plots.
figure(5);
clf(5);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	errorbar(Xs, mPost, 2*sPost, 2*sPost, 'k*-');
	plot(Xm, fmh, 'ko'); % We plot the measurement points.
else
	errorbar(Xs, mPost, 2*sPost, 2*sPost, 'bo-');
	plot(Xm, fmh, 'ro'); % We plot the measurement points.
end
axis([0,4,-1.5,2.5]);
if exportFigs ~= 0
	export_fig('SecondGPRegression1.png','-transparent');
end

% We define other data.
Xs = 0:0.01:4;
Xm = [1,2.5,3.7,0.1];
fmh = cos(Xm - 2)'; % These are the measurement values.

% We set up the (squared exponential) covariance matrix and related terms in exactly the same way.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% We apply GP regression in exactly the same way.
mPost = ms + Ksm/Kmm*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/Kmm*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up a plot in a new way, using a grey'ish area instead of error bars.
figure(5);
clf(5);
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
axis([0,4,-1.5,2.5]);
if exportFigs ~= 0
	export_fig('SecondGPRegression2.png','-transparent');
end

%% Figure 2.6.
disp('Creating Figure 2.6.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfm = 0.1; % This is the output noise scale.
Xs = 0:0.01:4; % These are the trial points.
Xm = [1,2.5,3.7,0.1]; % These are the measurement points.
fmh = cos(Xm - 2)' + sfm*randn(size(Xm))'; % These are the measurement values, corrupted by noise.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(6);
clf(6);
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
axis([0,4,-1.5,2.5]);
if exportFigs ~= 0
	export_fig('NoisyGPRegression.png','-transparent');
end

%% Figure 2.7.
disp('Creating Figure 2.7.');
% We first of all do the exact same thing as for figure 2.6.
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfm = 0.1; % This is the output noise scale.
Xs = 0:0.01:4; % These are the trial points.
Xm = [1,2.5,3.7,0.1]; % These are the measurement points.
fmh = cos(Xm - 2)' + sfm*randn(size(Xm))'; % These are the measurement values, corrupted by noise.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(7);
clf(7);
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
axis([0,4,-1.5,2.5]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(5, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
% sample1 = mvnrnd(mPost, SPost + eps*eye(ns))'; % We could have also used this, which is the equivalent of the previous two lines, but the mvnrnd function gives numerical problems more quickly than the cholesky function.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('GPSamples.png','-transparent');
end

%% Figure 2.8.
disp('Creating Figure 2.8.');
% We define data.
lf = 1; % This is the output length scale.
lx = [1;0.5]; % These are the input length scales.
sfm = 0.02; % This is the output noise scale.
xMin = [0;0]; % These are the minimums for the plot area.
xMax = [4;4]; % These are the maximums for the plot area.
rng(5, 'twister'); % I'm fixing Matlab's random number generator again.
nm = 20; % This is the number of measurement points which we will use.
Xm = repmat(xMin,1,nm) + rand(2,nm).*(repmat(xMax,1,nm) - repmat(xMin,1,nm));
fmh = (cos(Xm(1,:) - 2) + cos(2*(Xm(2,:) - 2)))' + sfm*randn(1,nm)'; % These are the measurement values, corrupted by noise.
rng('shuffle'); % And we unfix the random number generator again.

% We set up the trial points.
nsPerDimension = 51; % This is the number of trial points per dimension.
ns = nsPerDimension^2; % This is the total number of trial points.
[x1Mesh,x2Mesh] = meshgrid(linspace(xMin(1),xMax(1),nsPerDimension),linspace(xMin(2),xMax(2),nsPerDimension));
Xs = [reshape(x1Mesh,1,ns); reshape(x2Mesh,1,ns)];

% We now set up the (squared exponential) covariance matrix and related terms.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(permute(X,[3,2,1]),n,1) - repmat(permute(X,[2,3,1]),1,n); % This is matrix containing differences between input points. We have rearranged things so that indices 1 and 2 represent the numbers of vectors, while index 3 represents the element within the vector.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[3,2,1]),n,n), 3)); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

% And then we plot the result.
figure(8);
clf(8);
hold on;
grid on;
sDown = surface(x1Mesh, x2Mesh, mPost - 2*sPost);
set(sDown,'FaceAlpha',0.3);
set(sDown,'LineStyle','none');
sUp = surface(x1Mesh, x2Mesh, mPost + 2*sPost);
set(sUp,'FaceAlpha',0.3);
set(sUp,'LineStyle','none');
sMid = surface(x1Mesh, x2Mesh, mPost);
set(sMid,'FaceAlpha',0.8);
if useColor == 0
	set(sDown,'FaceColor',[0.2,0.2,0.2]);
	set(sUp,'FaceColor',[0.2,0.2,0.2]);
	set(sMid,'FaceColor',[0.2,0.2,0.2]);
	scatter3(Xm(1,:), Xm(2,:), fmh, 'ko', 'filled');
else
	set(sDown,'FaceColor',[0,0,0.8]);
	set(sUp,'FaceColor',[0,0,0.8]);
	set(sMid,'FaceColor',[0,0,0.8]);
	scatter3(Xm(1,:), Xm(2,:), fmh, 'ro', 'filled');
end
xlabel('Input 1');
ylabel('Input 2');
zlabel('Output');
view([118,10]);
if exportFigs ~= 0
	export_fig('MultiInputGPRegression.png','-transparent');
end


%% Figure 2.9.
disp('Creating Figure 2.9.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfm = 0.1; % This is the output noise scale.
Xs = 0:0.01:4; % These are the trial points.
Xm = [1,2.5,3.7,0.1]; % These are the measurement points.
fmh = cos(Xm - 2)' + sfm*randn(size(Xm))'; % These are the measurement values, corrupted by noise.

% We now set up the (squared exponential) covariance matrix and related terms.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(9);
clf(9);
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
axis([0,4,-1.5,2.5]);
if exportFigs ~= 0
	export_fig('OriginalGP.png','-transparent');
end

% We now take derivatives of the posterior distribution.
d2Kss = Kss.*(1/lx^2 - (diff(nm+1:end,nm+1:end).^2/lx^4)); % This is the derivative d^2 k(x,x') / dx dx' for the trial points.
dKms = Kms.*diff(1:nm,nm+1:end)/lx^2; % This is the derivative dk(x,x') / dx' where the first input is the set of measurement points and the second input is the set of trial points.
dKsm = dKms'; % This is the derivative dk(x,x') / dx where the first input is the set of trial points and the second input is the set of measurement points.
mdPost = -dKsm/(Kmm + Sfm)*(fmh - mm); % These are the posterior mean of the derivative.
SdPost = d2Kss - dKsm/(Kmm + Sfm)*dKms; % These are the posterior covariance of the derivative.
sdPost = sqrt(diag(SdPost)); % These are the posterior standard deviations of the derivative.

% We set up the GP plot.
figure(9);
clf(9);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mdPost-2*sdPost; flipud(mdPost+2*sdPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mdPost-sdPost; flipud(mdPost+sdPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mdPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mdPost-2*sdPost; flipud(mdPost+2*sdPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mdPost-sdPost; flipud(mdPost+sdPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mdPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([0,4,-2,2]);
if exportFigs ~= 0
	export_fig('DerivativeGP.png','-transparent');
end

%% Figure 2.10.
disp('Creating Figure 2.10.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfm = 0.01; % This is the output noise scale.
sdfh = 0.01; % This is the output derivative noise scale.
Xs = 0:0.01:4; % These are the trial points.
Xm = [1,2.5,3.7,0.1,3.2]; % These are the measurement points.
useValueM = logical([1,0,1,0,0]); % This is an array indicating whether we want to use the function value for the given point. We also turn it into logicals, so Matlab allows us to use it as indices.
useDerivativeM = logical([1,1,0,1,0]); % This is an array indicating whether we want to use the derivative for the given point.
fmh = [(useValueM.*cos(Xm - 2))' + sfm*randn(size(Xm))'; (useDerivativeM.*(-sin(Xm - 2)))' + sdfh*randn(size(Xm))']; % These are the measurement values, corrupted by noise. We first have the measurement values and then the derivative values.

% We now set up a covariance matrix. Or actually, we set up four, based on whether we are using derivatives or values.
nm = size(Xm,2); % This is the number of measurement points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
Kvv = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kdv = Kvv.*(diff/lx^2); % This is the covariance between the function derivatives and the function values.
Kvd = Kvv.*(-diff/lx^2); % This is the covariance between the function values and the function derivatives.
Kdd = Kvv.*(repmat(lx^2,n,n) - diff.^2)/lx^2;
Kmm = [Kvv(1:nm,1:nm),Kvd(1:nm,1:nm);Kdv(1:nm,1:nm),Kdd(1:nm,1:nm)];
Kms = [Kvv(1:nm,nm+1:end);Kdv(1:nm,nm+1:end)];
Ksm = Kms';
Kss = Kvv(nm+1:end,nm+1:end);
Sfm = diag([sfm^2*(ones(1,nm) + (1 - useValueM)*1e6^2),sdfh^2*(ones(1,nm) + (1 - useDerivativeM)*1e6^2)]); % This is the noise covariance matrix. Here, we set certain covariances to the correct value. We set other covariances to infinity (or a very large value) to indicate that we don't use those.
mm = zeros(2*nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We calculate some line properties for the derivative lines.
derivativeLineLength = 0.2; % We set the length of the derivative lines.
mPostM = (inv(Kmm) + inv(Sfm))\(Kmm\mm + Sfm\fmh); % We calculate the posterior mean of the measurement points.
lineX = [Xm' - derivativeLineLength/2, Xm' + derivativeLineLength/2]; % These are the starting and ending x-coordinates of the derivative lines.
lineY = [useValueM'.*fmh(1:nm) + (1 - useValueM)'.*mPostM(1:nm) - fmh(nm+1:2*nm)*derivativeLineLength/2, useValueM'.*fmh(1:nm) + (1 - useValueM)'.*mPostM(1:nm) + fmh(nm+1:2*nm)*derivativeLineLength/2]; % These are the starting and ending y-coordinates of the derivative lines. As base y-coordinate, we use the measured value fmh wherever it is used, and the posterior value from mPostM otherwise.

% We set up the GP plot.
figure(10);
clf(10);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm(useValueM), fmh(useValueM), 'ko'); % We plot the measurement points.
	for i = 1:size(lineX,1) % And we plot the derivative lines.
		if useDerivativeM(i) % We only draw the lines that we have actually used.
			plot(lineX(i,:), lineY(i,:), 'k-', 'LineWidth', 2);
		end
	end
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm(useValueM), fmh(useValueM), 'ro'); % We plot the measurement points.
	for i = 1:size(lineX,1) % And we plot the derivative lines.
		if useDerivativeM(i) % We only draw the lines that we have actually used.
			plot(lineX(i,:), lineY(i,:), 'r-', 'LineWidth', 1);
		end
	end
end
axis([0,4,-1.5,2.5]);
if exportFigs ~= 0
	export_fig('UsingDerivativeData.png','-transparent');
end