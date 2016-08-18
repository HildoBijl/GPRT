% This file contains all the scripts for Chapter 6 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter6 from the Matlab command.

% We set up the workspace, ready for executing scripts.
% clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.
addpath('../Tools'); % This is for a few useful add-on functions, like the logdet function.

% We define colors.
black = [0 0 0];
white = [1 1 1];
if useColor == 0
	red = [0 0 0];
	green = [0.6 0.6 0.6];
	blue = [0.2 0.2 0.2];
	yellow = [0.4 0.4 0.4];
	grey = [0.8 0.8 0.8];
	brown = [0.95 0.95 0.95];
else
	red = [0.8 0 0];
	green = [0 0.4 0];
	blue = [0 0 0.8];
	yellow = [0.6 0.6 0];
	grey = [0.8 0.8 1];
	brown = [0.45 0.15 0.0];
end

% We generate measurements for an example GP which we will use.
nm = 20; % This is the number of measurement points.
ns = 301; % This is the number of plot (trial) points.
xMin = -3; % This is the minimum input.
xMax = 3; % This is the maximum input.
fmMin = 0; % This is the minimum value which f_{max} can be.
fmMax = 1.2; % This is the maximum value which f_{max} can be.
sfm = 0.3;
rng(227, 'twister'); % We fix Matlab's random number generator, so we get the same measurement points as in the thesis plots.
Xm = xMin + rand(1,nm)*(xMax - xMin);
fm = (cos(3*Xm) - Xm.^2/9 + Xm/6)';
fmh = fm + sfm*randn(nm,1);
Xs = linspace(xMin, xMax, ns);
fs = (cos(3*Xs) - Xs.^2/9 + Xs/6)';
dx = (xMax - xMin)/(ns - 1); % This is the distance between two trial points.
fm = linspace(fmMin, fmMax, ns); % This is the x-axis scale for the plots of the maximum value distribution.
hf = 0.04; % This is the length scale for the kernel density estimation process of the maximum value distribution.

% We set up a Gaussian process to approximate the measurements, giving us the GP for our examples.
lf = 1; % This is the output length scale.
lx = 0.6; % This is the input length scale.
X = [Xm,Xs];
n = size(X,2);
diff = repmat(X,n,1) - repmat(X',1,n); % This is the matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nm,ns],[nm,ns]);
Kmm = KDivided{1,1};
Kms = KDivided{1,2};
Ksm = KDivided{2,1};
Kss = KDivided{2,2};
mm = zeros(nm,1); % This is the prior mean vector of the measurement points.
ms = zeros(ns,1); % This is the prior mean vector of the trial points.
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

%% Figure 6.1.
disp('Creating Figure 6.1.');

% We generate samples from the posterior distribution of the trial points.
rng(37, 'twister'); % We fix Matlab's random number generator, so that it is guaranteed to find samples with very different maximums, illustrating the point that I want to make.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
[m1,i1] = max(sample1);
[m2,i2] = max(sample2);
[m3,i3] = max(sample3);

% We plot the GP and the samples.
figure(1);
clf(1);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xs, sample1, '-', 'Color', green); % We draw the three different samples.
plot(Xs, sample2, '-', 'Color', yellow);
plot(Xs, sample3, '-', 'Color', brown);
plot(Xs(i1), sample1(i1), 'x', 'Color', green); % We use a cross to indicate the maximums.
plot(Xs(i2), sample2(i2), 'x', 'Color', yellow);
plot(Xs(i3), sample3(i3), 'x', 'Color', brown);
% plot(Xs, fs, '-', 'LineWidth', 1, 'Color', black); % This plots the true function which we are approximating.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
axis([xMin,xMax,-2,1.5]);
if exportFigs ~= 0
	export_fig('GPExampleWithSamples.png','-transparent');
end

%% Figure 6.2.
disp('Creating Figure 6.2.');

% We generate a lot of samples and find their maximums. Then we use this to find the maximum distribution.
nSamples = 1e5; % This is the number of sample functions we will examine.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
maxCounter = zeros(1,ns); % We will use this array to count how many times a specific point is the maximum.
maxValDist = zeros(1,ns); % We will use this array to set up a probability density function of the maximum value. We will not use/plot that here in this block, but we will use it later in Figure 6.9.
for i = 1:nSamples
	sample = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian process.
	[val,ind] = max(sample); % We find out where the maximum occurs. That is, at which of the ns trial points.
	maxCounter(ind) = maxCounter(ind) + 1; % We increment the counter for this trial point.
	maxValDist = maxValDist + 1/sqrt(det(2*pi*hf^2))*exp(-1/2*(fm - val).^2/hf^2); % We use Kernel density estimation to set up this distribution. It's a trick to make the plot more smooth and hence more close to the actual probability density function.
end
maxDist = maxCounter/nSamples/dx; % This is the maximum distribution. It is a probability density function, meaning its integral adds up to one.
maxValDist = maxValDist/nSamples; % This is the distribution of the maximum value.

% Then we plot the maximum distribution.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Maximum probability density function');
plot(Xs, maxDist, '-', 'Color', blue);
axis([xMin,xMax,0,2.5]);
if exportFigs ~= 0
	export_fig('GPMaximumDistribution.png','-transparent');
end

% % Finally we save it, since we may need it for later applications.
% save('TrueMaximumDistribution','Xs','maxDist');
% save('TrueMaximumValueDistribution','fm','maxValDist');

%% Figure 6.3.
disp('Creating Figure 6.3.');

% We take derivatives of the posterior distribution.
d2Kss = Kss.*(1/lx^2 - (diff(nm+1:end,nm+1:end).^2/lx^4)); % This is the derivative d^2 k(x,x') / dx dx' for the trial points.
dKms = Kms.*diff(1:nm,nm+1:end)/lx^2; % This is the derivative dk(x,x') / dx' where the first input is the set of measurement points and the second input is the set of trial points.
dKsm = dKms'; % This is the derivative dk(x,x') / dx where the first input is the set of trial points and the second input is the set of measurement points.
mdPost = zeros(ns,1) + dKsm/(Kmm + Sfm)*(fmh - mm); % These are the posterior mean of the derivative.
SdPost = d2Kss - dKsm/(Kmm + Sfm)*dKms; % These are the posterior covariance of the derivative.
sdPost = sqrt(diag(SdPost)); % These are the posterior standard deviations of the derivative.

% We set up the GP plot.
figure(3);
clf(3);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mdPost-2*sdPost; flipud(mdPost+2*sdPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mdPost-sdPost; flipud(mdPost+sdPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mdPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
if exportFigs ~= 0
	export_fig('DerivativeGP.png','-transparent');
end

% At each point the function value derivative has a Gaussian distribution. So for each function value derivative we can calculate the probability that it is zero.
zeroDerivProb = 1./sqrt(2*pi*sdPost.^2).*exp(-1/2*mdPost.^2./sdPost.^2);
zeroDerivProb = zeroDerivProb/sum(zeroDerivProb)/dx;
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel('Zero derivative probability');
plot(Xs, zeroDerivProb, '-', 'Color', blue);
if exportFigs ~= 0
	export_fig('ZeroDerivativeProbability.png','-transparent');
end

% We now set up the distribution of the second derivative, given our measurements and given that the first derivative at that exact point is zero.
negativeSecondDerivProb = zeros(ns,1);
for i = 1:ns % We walk through each individual point, noting that the derivative there is zero and calculating the posterior distribution of the second derivative.
	d2Ksm = Ksm(i,:).*(-1/lx^2 + (diff(nm+i,1:nm).^2/lx^4)); % This is the derivative d^2 k(x,x') / dx^2 where the first input is the set of trial points and the second input is the set of measurement points.
	d3Kss = Kss(i,i).*(3*diff(nm+i,nm+i)/lx^4 - diff(nm+i,nm+i).^3/lx^6); % This is the derivative d^3 k(x,x') / dx^2 dx' where both inputs are the trial set.
	d4Kss = Kss(i,i).*(3/lx^4 - 6*diff(nm+i,nm+i).^2/lx^6 + diff(nm+i,nm+i).^4/lx^8); % This is the derivative d^4 k(x,x') / dx^2 dx'^2 where both inputs are the trial set.
	mddPost = zeros(1,1) + [d2Ksm,d3Kss]/[Kmm + Sfm,dKms(:,i);dKsm(i,:),d2Kss(i,i)]*[fmh - mm;0]; % Here we apply the regression. Note that we have set up a joint distribution of both f (at the position of the measurements), df/dx (at the position of the current point) and d^2f/dx^2 (at the position of the current point). We know the first two (albeit with some possible noise) and apply regression to find the third.
	SddPost = d4Kss - [d2Ksm,d3Kss]/[Kmm + Sfm,dKms(:,i);dKsm(i,:),d2Kss(i,i)]*[d2Ksm';d3Kss'];
	sddPost = sqrt(diag(SddPost));
	negativeSecondDerivProb(i) = normcdf(0,mddPost,sddPost);
end
maximumProb = negativeSecondDerivProb.*zeroDerivProb; % We multiply the p(d^2f/dx^2 < 0|df/dx = 0) by p(df/dx = 0) for each point, according to the conditional probabilities.
maximumProb = maximumProb/sum(maximumProb)/dx; % We normalize the result to make sure it's a valid PDF whose integral equals one.

% We plot the outcome.
figure(5);
clf(5);
hold on;
grid on;
xlabel('Input');
ylabel('Maximum probability');
plot(Xs, maximumProb, '-', 'Color', blue);
if exportFigs ~= 0
	export_fig('LocalMaximumProbability.png','-transparent');
end

%% Figure 6.4.
disp('Creating Figure 6.4.');

% We set up bins.
nr = 20; % We define the number of rounds we apply.
np = 1e4; % We define the number of particles used.
bins = zeros(ns,nr+1); % We set up storage space for the number of bins.
bins(:,1) = floor(np/ns); % We divide the particles over the bins.
bins(1:mod(np,ns),1) = ceil(np/ns); % We give the first few bins an extra particle if it doesn't quite match.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it gives the plot shown in the thesis.

% We iterate over the number of rounds.
for i = 1:nr
	% We walk through all the bins.
	for j = 1:ns
		% For each particle, we make a comparison.
		for k = 1:bins(j,i)
			randomBin = ceil(rand(1,1)*ns); % We pick a new random bin.
			mut = mPost(j) - mPost(randomBin); % We set up the posterior distribution of f_j - f_r, with f_j being the current bin and f_r being the new random bin.
			St = SPost(j,j) + SPost(randomBin,randomBin) - 2*SPost(j,randomBin);
			sample = mut + sqrt(St)*randn(1,1); % This is a sample from f_j - f_r. If it is positive, then f_j > f_r and the particle can stay. If it is negative, then f_j < f_r and it should move to the new random bin.
			if sample >= 0
				bins(j,i+1) = bins(j,i+1) + 1;
			else
				bins(randomBin,i+1) = bins(randomBin,i+1) + 1;
			end
		end
	end
end

% Next, we calculate the limit distribution of the particles.
P = zeros(ns,ns);
for i = 1:ns
	for j = 1:ns
		mut = mPost(i) - mPost(j);
		Sigmat = SPost(i,i) + SPost(j,j) - SPost(i,j) - SPost(j,i);
		P(i,j) = erf(mut/sqrt(2*Sigmat))/2 + 1/2;
	end
	P(i,i) = 1/2;
end

% We calculate the comparison matrix and use it to find the limit distribution of the particles.
mat = diag(diag(ones(ns,ns)*P)) - P;
outcome = zeros(ns,1);
mat(end,:) = ones(1,ns); % We set the bottom row equal to ones.
outcome(end) = 1; % We set the bottom element of the outcome equal to one.
limitDist = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
limitDist = limitDist/dx; % We turn the result into a PDF.

% We plot the results.
figure(5);
clf(5);
hold on;
grid on;
xlabel('Input');
ylabel('Particle distribution');
for i = 1:nr+1
	distribution = bins(:,i)/np/dx;
	particleDistribution = plot(Xs, distribution, '-', 'Color', 0.8*(1 - i/(nr+1))*[1,1,1]);
end
axis([xMin,xMax,0,2.5]);

% We also add the true maximum distribution and the limit distribution to the plot.
limitDistribution = plot(Xs, limitDist, '-', 'Color', red, 'LineWidth', 1);
load('TrueMaximumDistribution');
trueDistribution = plot(Xs, maxDist, '-', 'Color', blue);
legend([trueDistribution, limitDistribution, particleDistribution], 'True distribution', 'Limit distribution', 'Particle distribution', 'Location', 'NorthWest');
if exportFigs ~= 0
	export_fig('ParticleDistribution.png','-transparent');
end

%% Example problem.
disp('Setting up the example problem.');

% We set up the distribution.
eps = 1e-6; % This is a very small number.
mup = [0;0;0];
S = [1,1,0;1,1+eps^2,0;0,0,1];

% We calculate the difference distribution for each of the points to find the true maximum distribution.
trueDist = zeros(3,1);
mut = repmat(mup(1),2,1) - mup(2:3);
St = repmat(S(1,1),2,2) - repmat(S(1,2:3),2,1) - repmat(S(2:3,1),1,2) + S(2:3,2:3);
trueDist(1) = mvncdf(mut,zeros(2,1),St);
mut = repmat(mup(3),2,1) - mup(1:2);
St = repmat(S(3,3),2,2) - repmat(S(3,1:2),2,1) - repmat(S(1:2,3),1,2) + S(1:2,1:2);
trueDist(3) = mvncdf(mut,zeros(2,1),St);
trueDist(2) = 1 - trueDist(1) - trueDist(3);
disp('The true maximum distribution is given by');
disp(trueDist);

% We calculate the comparison matrix and use it to find the limit distribution if we would use particles.
P = zeros(3,3);
for i = 1:3
	for j = 1:3
		ind1 = i;
		ind2 = j;
		mut = mup(ind1) - mup(ind2);
		St = S(ind1,ind1) - S(ind1,ind2) - S(ind2,ind1) + S(ind2,ind2);
		P(i,j) = erf(mut/sqrt(2*St))/2 + 1/2;
	end
	P(i,i) = 1/2; % It does not matter to what value we set it, but if we do not set it, it will remain NaN and things will go awry.
end
mat = P - diag(diag(ones(3,3)*P));
outcome = zeros(3,1);
mat(end,:) = ones(1,3); % We set the bottom row equal to ones.
outcome(end) = 1; % We set the bottom element of the outcome equal to one.
limitDist = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
disp('The limit distribution is given by');
disp(limitDist);

%% Figure 6.5.
disp('Creating Figure 6.5.');

% We make some definitions.
np1 = 20; % This is the number of particles we use.
np2 = 1e3; % This is the other number of particles we use.
x0Min = 0.6; % This is the minimum of the initial distribution.
x0Max = 1.4; % This is the maximum of the initial distribution.
nPlot = 501; % This is the number of plot points we will use.
xsMin = -0.5; % This is the minimum of the plot.
xsMax = 1.5; % This is the maximum of the plot.
xs = linspace(xsMin, xsMax, nPlot);
dxPlot = (xsMax - xsMin)/nPlot; % This is the distance between two plot points.
sx = 0.1; % This is the process noise present in the system.
sysA = 0.8; % The discrete-time system A matrix.
rng(1, 'twister'); % We fix Matlab's random number generator, so we get the same measurement points as in the thesis plots.

% We set up the PDF of the initial distribution and plot it.
initialDist = ((xs <= x0Max).*(xs > x0Min))/(x0Max - x0Min);
figure(6);
clf(6);
hold on;
grid on;
xlabel('x_k');
ylabel('p(x_k)');
plot(xs, initialDist, '-', 'Color', blue);
if exportFigs ~= 0
	export_fig('InitialStateDistribution.png','-transparent');
end

% We calculate the desired posterior distribution. Since the new state x_{k+1} is the summation of the uniform initial distribution and the Gaussian noise, we can find its distribution by
% calculating the convolution between these two distributions. (This is not explained in the thesis. You'll just have to trust me on this.)
desPostDist = (normcdf((sysA*x0Max - xs)/sx) - normcdf((sysA*x0Min - xs)/sx))/sysA/(x0Max - x0Min);

% We generate a number of particles and calculate the next particles.
xp1 = x0Min + (x0Max - x0Min)*rand(np1,1);
xp1New = sysA*xp1 + sx*randn(np1,1);
xp2 = x0Min + (x0Max - x0Min)*rand(np2,1);
xp2New = sysA*xp2 + sx*randn(np2,1);

% We set up a plot where we represent the particles as delta points. Actually delta points are infinitely high so we cannot plot those. Instead, I will use very narrow block functions with a width dx of one plot point.
postDist1 = zeros(size(initialDist));
postDist2 = zeros(size(initialDist));
for i = 1:np1
	xcMin = xp1New(i) - dxPlot/2;
	xcMax = xp1New(i) + dxPlot/2;
 	postDist1 = postDist1 + (1/np1)*((xs <= xcMax).*(xs > xcMin))/dxPlot;
end
for i = 1:np2
	xcMin = xp2New(i) - dxPlot/2;
	xcMax = xp2New(i) + dxPlot/2;
 	postDist2 = postDist2 + (1/np2)*((xs <= xcMax).*(xs > xcMin))/dxPlot;
end
figure(7);
clf(7);
hold on;
grid on;
xlabel('x_{k+1}');
ylabel('p(x_{k+1})');
% plot(xs, postDist2, '-', 'Color', red);
plot(xs, postDist1, '-', 'Color', blue);
% plot(xs, desPostDist, '-', 'Color', green);
if exportFigs ~= 0
	export_fig('NextStateDistributionThroughDelta.png','-transparent');
end

% We now represent the particles as blocks that are a bit wider than just dx.
blockWidth = 0.1;
postDist1 = zeros(size(initialDist));
postDist2 = zeros(size(initialDist));
for i = 1:np1
	xcMin = xp1New(i) - blockWidth/2;
	xcMax = xp1New(i) + blockWidth/2;
 	postDist1 = postDist1 + (1/np1)*((xs <= xcMax).*(xs > xcMin))/blockWidth;
end
for i = 1:np2
	xcMin = xp2New(i) - blockWidth/2;
	xcMax = xp2New(i) + blockWidth/2;
 	postDist2 = postDist2 + (1/np2)*((xs <= xcMax).*(xs > xcMin))/blockWidth;
end
figure(8);
clf(8);
hold on;
grid on;
xlabel('x_{k+1}');
ylabel('p(x_{k+1})');
plot(xs, postDist1, '-', 'Color', blue);
plot(xs, postDist2, '-', 'Color', red);
plot(xs, desPostDist, '-', 'Color', green);
legend('Approximation with n_p = 20','Approximation with n_p = 1000','Analytical distribution','Location','NorthWest');
if exportFigs ~= 0
	export_fig('NextStateDistributionThroughBlock.png','-transparent');
end

% And now we represent the particles through a Gaussian kernel.
GaussianWidth = 0.05;
postDist1 = zeros(size(initialDist));
postDist2 = zeros(size(initialDist));
for i = 1:np1
 	postDist1 = postDist1 + (1/np1)*1/sqrt(2*pi*GaussianWidth^2)*exp(-1/2*(xp1New(i) - xs).^2/GaussianWidth^2);
end
for i = 1:np2
 	postDist2 = postDist2 + (1/np2)*1/sqrt(2*pi*GaussianWidth^2)*exp(-1/2*(xp2New(i) - xs).^2/GaussianWidth^2);
end
figure(9);
clf(9);
hold on;
grid on;
xlabel('x_{k+1}');
ylabel('p(x_{k+1})');
plot(xs, postDist1, '-', 'Color', blue);
plot(xs, postDist2, '-', 'Color', red);
plot(xs, desPostDist, '-', 'Color', green);
legend('Approximation with n_p = 20','Approximation with n_p = 1000','Analytical distribution','Location','NorthWest');
if exportFigs ~= 0
	export_fig('NextStateDistributionThroughGaussian.png','-transparent');
end

%% Figure 6.6.
disp('Creating Figure 6.6.');

% We define the particles.
x = 1:5; % These are the particle values.
w = [0.1, 0.4, 1, 1.5, 2]; % These are the particle weights.
np = length(x); % This is the number of particles.

% We set up data for the cumulative particle weight plot.
xs = 0:0.001:6;
ys = zeros(size(xs));
for i = 1:np
	indices = xs > x(i);
	ys(indices) = ys(indices) + w(i);
end

% We make the cumulative weight plot.
figure(10);
clf(10);
hold on;
grid on;
xlabel('Weight number');
ylabel('Cumulative particle weight');
plot(xs, ys, '-', 'Color', blue);
if exportFigs ~= 0
	export_fig('CumulativeParticleWeight.png','-transparent');
end

% We now set up data for the multinomial resampling idea and make that plot.
rng(7, 'twister'); % We fix Matlab's random number generator, so we get the same measurement points as in the thesis plots.
newParticleIndices = rand(np,1)*np;
figure(11);
clf(11);
hold on;
grid on;
xlabel('Weight number');
ylabel('Cumulative particle weight');
plot(xs, ys, '-', 'Color', blue);
for i = 1:np
	plot([min(xs),max(xs)], [newParticleIndices(i),newParticleIndices(i)], '-', 'Color', green);
end
if exportFigs ~= 0
	export_fig('MultinomialResampling.png','-transparent');
end

% Next, we set up data for the stratified resampling idea and make that plot.
rng(6, 'twister'); % We fix Matlab's random number generator, so we get the same measurement points as in the thesis plots.
figure(12);
clf(12);
hold on;
grid on;
xlabel('Weight number');
ylabel('Cumulative particle weight');
for i = 1:np
	if mod(i,2) == 0 % We alternate regular grey and light grey colors.
		patch([min(xs),max(xs),max(xs),min(xs)],[i-1,i-1,i,i], 1, 'FaceColor', white, 'EdgeColor', 'none'); % This is the grey area in the plot.
	else
		patch([min(xs),max(xs),max(xs),min(xs)],[i-1,i-1,i,i], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
	end
end
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
for i = 1:np
	newParticleIndices(i) = rand(1,1) + i - 1;
	plot([min(xs),max(xs)], [newParticleIndices(i),newParticleIndices(i)], '-', 'Color', green);
end
plot(xs, ys, '-', 'Color', blue);
if exportFigs ~= 0
	export_fig('StratifiedResampling.png','-transparent');
end

% Finally, we set up systematic resampling.
rng(3, 'twister'); % We fix Matlab's random number generator, so we get the same measurement points as in the thesis plots.
newParticleIndices = (1:np) - rand(1,1);
figure(13);
clf(13);
hold on;
grid on;
xlabel('Weight number');
ylabel('Cumulative particle weight');
plot(xs, ys, '-', 'Color', blue);
for i = 1:np
	plot([min(xs),max(xs)], [newParticleIndices(i),newParticleIndices(i)], '-', 'Color', green);
end
if exportFigs ~= 0
	export_fig('SystematicResampling.png','-transparent');
end

%% Figure 6.7.
disp('Creating Figure 6.7.');

% We set up a GP with only a very limited number of measurements, so we have lots of uncertainty.
nmUsed = 8;
mPostUnc = ms + Ksm(:,1:nmUsed)/(Kmm(1:nmUsed,1:nmUsed) + Sfm(1:nmUsed,1:nmUsed))*(fmh(1:nmUsed) - mm(1:nmUsed)); % This is the posterior mean vector.
SPostUnc = Kss - Ksm(:,1:nmUsed)/(Kmm(1:nmUsed,1:nmUsed) + Sfm(1:nmUsed,1:nmUsed))*Kms(1:nmUsed,:); % This is the posterior covariance matrix.
sPostUnc = sqrt(diag(SPostUnc)); % These are the posterior standard deviations.

% We plot the GP which we had previously.
figure(14);
clf(14);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPostUnc-2*sPostUnc; flipud(mPostUnc+2*sPostUnc)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPostUnc-sPostUnc; flipud(mPostUnc+sPostUnc)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPostUnc, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
% plot(Xs, fs, '-', 'LineWidth', 1, 'Color', black); % This plots the true function which we are approximating.
plot(Xm(1:nmUsed), fmh(1:nmUsed), 'o', 'Color', red); % We plot the measurement points.
axis([xMin,xMax,-2,2]);
if exportFigs ~= 0
	export_fig('GPWithFewerMeasurements.png','-transparent');
end

% We calculate the various acquisition functions.
fsh = max(mPostUnc); % This is \hat{f}^*.
xi = 0.2; % This is the exploration parameter for the PI and EI acquisition functions.
z0 = (mPostUnc - fsh)./sPostUnc;
zd = (mPostUnc - fsh - xi)./sPostUnc;
PI0 = normcdf(z0);
PId = normcdf(zd);
EI0 = (mPostUnc - fsh).*normcdf(z0) + sPostUnc.*normpdf(z0);
EId = (mPostUnc - fsh - xi).*normcdf(zd) + sPostUnc.*normpdf(zd);
EV = mPostUnc;
UCB1 = mPostUnc + 1*sPostUnc;
UCB2 = mPostUnc + 2*sPostUnc;

% We scale some acquisition functions to make them appear well together in the same plot. Note that this does not affect the position of their maximum.
% PI = PI/mean(PI); % This scales the PI AF to be one on average.
% EI = EI/mean(EI); % This scales the EI AF to be one on average.
% EI = EI/mean(EI)*mean(PI); % This scales the EI AF to be the same as the PI AF on average.

% We find the positions of the maximums of the acquisition functions.
[PI0max,PI0ind] = max(PI0);
[EI0max,EI0ind] = max(EI0);
[PIdmax,PIdind] = max(PId);
[EIdmax,EIdind] = max(EId);
[EVmax,EVind] = max(EV);
[UCB1max,UCB1ind] = max(UCB1);
[UCB2max,UCB2ind] = max(UCB2);

% We plot the various acquisition functions.
figure(15);
clf(15);
hold on;
grid on;
xlabel('Input');
ylabel('Acquisition function value');
plot(Xs, UCB2, '-', 'Color', yellow);
plot(Xs, UCB1, '-', 'Color', blue);
plot(Xs, EV, '-', 'Color', red);
plot(Xs(UCB2ind), UCB2max, 'x', 'Color', yellow);
plot(Xs(UCB1ind), UCB1max, 'x', 'Color', blue);
plot(Xs(EVind), EVmax, 'x', 'Color', red);
legend('UCB AF (\kappa=2)','UCB AF (\kappa=1)','EV AF (\kappa=0)');
if exportFigs ~= 0
	export_fig('EVandUCBAF.png','-transparent');
end

% We plot the various acquisition functions.
figure(16);
clf(16);
hold on;
grid on;
xlabel('Input');
ylabel('Acquisition function value');
plot(Xs, PI0, '-', 'Color', green);
plot(Xs, PId, '-', 'Color', yellow);
plot(Xs, EI0, '-', 'Color', blue);
plot(Xs, EId, '-', 'Color', red);
plot(Xs(PI0ind), PI0max, 'x', 'Color', green);
plot(Xs(PIdind), PIdmax, 'x', 'Color', yellow);
plot(Xs(EI0ind), EI0max, 'x', 'Color', blue);
plot(Xs(EIdind), EIdmax, 'x', 'Color', red);
legend('PI AF (\xi=0)',['PI AF (\xi=',num2str(xi),')'],'EI AF (\xi=0)',['EI AF (\xi=',num2str(xi),')'],'Location','NorthWest');
if exportFigs ~= 0
	export_fig('PIandEIAF.png','-transparent');
end

%% Figures 6.8. and 6.9.
disp('Creating Figure 6.8.');

% We define parameters.
nr = 10; % We define the number of rounds we apply.
np = 1e4; % We define the number of particles used.
alpha = 0.5; % This is the part of the time we sample a challenger from the current belief of the maximum distribution. A high value of alpha (near 1) speeds up convergence, but distorts the results slightly. For larger problems it is wise to start with a high value of alpha but decrease it towards zero as the algorithm converges.
h = 0.04; % This is the length scale of the Gaussian kernel we will use in the kernel density estimation process. It has been tuned to be the smallest number to still give smooth plots.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it gives the plot shown in the thesis.

% We initialize the particles.
particles = xMin + (xMax - xMin)*rand(1,np); % We set up particles in random locations.
weights = ones(np,1); % We initialize the weights.
% Next, we generate the victory function values. We initialize them according to the distribution of the function value at the particle input points.
diff = repmat(particles,nm,1) - repmat(Xm',1,np); % This is the matrix containing differences between input points.
Kmp = lf^2*exp(-1/2*diff.^2/lx^2);
Kpm = Kmp';
Kpp = lf^2*ones(np,1); % Usually this matrix is diagonal, but given that we have a lot of particles, that would be a too large matrix.
KpmDivKmm = Kpm/(Kmm + Sfm);
mup = KpmDivKmm*(fmh - mm); % These are the mean values of the Gaussian process at all the particle points.
Sigma = Kpp + sum(KpmDivKmm.*Kpm,2); % These are the variances of the Gaussian process at the particle points. We have calculated only the diagonal elements of the covariance matrix here, because we do not need the other elements.
sigma = sqrt(Sigma);
values = mup + sqrt(Sigma).*(randn(np,1)); % We take a random sample from the resulting distribution.

% We set up a storage parameter for the maximum distributions, and we set up the PDF for the first one.
pMax = zeros(nr+1, ns);
for i = 1:np
	pMax(1,:) = pMax(1,:) + weights(i)*1/sqrt(det(2*pi*h^2))*exp(-1/2*(Xs - particles(i)).^2/h^2);
end
pMax(1,:) = pMax(1,:)/sum(weights);

% We iterate over the number of challenge rounds.
for round = 1:nr
	% We start by applying systematic resampling. (Yes, this is quite useless in the first round, but we ignore that tiny detail and do it anyway.)
	oldParticles = particles; % We store the old particles, so we can override the particles matrix during the process of resampling.
	oldValues = values;
	oldWeights = weights;
	wCum = cumsum(oldWeights); % These are the cumulative weights.
	wSum = wCum(end); % This is the sum of all the weights.
	stepSize = wSum/np; % We calculate the step size based on the sum of all the weights.
	val = rand(1,1)*stepSize; % We pick a random number for the algorithm.
	oldPCounter = 1; % We use two counters in the process. This first one keeps track of which old particle we are at.
	newPCounter = 1; % This second counter keeps track of which new particle index we are at.
	while newPCounter <= np % We iterate until we have added np new particles.
		while wCum(oldPCounter) < val + (newPCounter-1)*stepSize % We iterate through the particles until we find the one which we should be adding particles of.
			oldPCounter = oldPCounter + 1;
		end
		while wCum(oldPCounter) >= val + (newPCounter-1)*stepSize % We keep adding this particle to the new set of particles until we have added enough.
			particles(newPCounter) = oldParticles(oldPCounter);
			weights(newPCounter) = 1;
			values(newPCounter) = oldValues(oldPCounter);
			newPCounter = newPCounter + 1;
		end
	end
	
	% We now create challengers according to the specified rules.
	sampleFromMaxDist = (rand(1,np) < alpha); % We determine which challengers we will pick from the current belief of the maximum distribution, and which challengers we pick randomly.
	randomPoints = rand(1,np)*(xMax-xMin)+xMin; % We select random challengers. (We do this for all particles and then discard the ones which we do not need.)
	indices = ceil(rand(np,1)*np); % We pick the indices of the champion particles we will use to generate challengers from. This line only works if all the particles have a weight of one, which is the case since we have just resampled. Otherwise, we should use randsample(np,np,true,weights);
	deviations = randn(1,np)*h; % To apply the Gaussian kernel to the selected champions, we need to add a Gaussian parameter to the champion particles. We set that one up here.
	challengers = (1-sampleFromMaxDist).*randomPoints + sampleFromMaxDist.*(particles(indices) + deviations); % We finalize the challenger points, picking random ones where applicable and sampled from the maximum distribution in other cases.

	% We now set up the covariance matrices and calculate some preliminary parameters.
	diff = repmat([particles,challengers],nm,1) - repmat(Xm',1,2*np); % This is the matrix containing differences between input points.
	Kmp = lf^2*exp(-1/2*diff.^2/lx^2);
	Kpm = Kmp';
	KpmDivKmm = Kpm/(Kmm + Sfm);
	mup = KpmDivKmm*(fmh - mm); % These are the mean values of the Gaussian process at all the particle points.

	% We calculate the mean and covariance for each combination of challenger and challenged point. Then we sample \hat{f} and look at the result.
	oldParticles = particles;
	oldValues = values;
	oldWeights = weights;
	for i = 1:np
		mupc = mup([i,i+np]); % This is the current mean vector.
		diff = repmat([particles(i),challengers(i)],2,1) - repmat([particles(i),challengers(i)]',1,2);
		Kppc = lf^2*exp(-1/2*diff.^2/lx^2);
		Sigmac = Kppc - KpmDivKmm([i,i+np],:)*Kmp(:,[i,i+np]); % This is the current covariance matrix.
		try % We use a try-catch-block here because sometimes numerical errors may occur.
			fHat = mupc + chol(Sigmac)'*randn(2,1);
			if fHat(2) > fHat(1) % Has the challenger won?
				particles(i) = challengers(i);
				values(i) = fHat(2);
				q = 1/(xMax-xMin); % This is the probability density function value of q(x).
				qp = (1-alpha)*q + alpha*1/sqrt(det(2*pi*h^2))*exp(-1/2*(challengers(i) - oldParticles(indices(i)))^2/h^2); % This is the sampling probability density function given that we have selected the champion particle from the indices vector in the sampling process.
				weights(i) = q/qp;
			end
		catch
			% Apparently challengerPoints(i) and points(i) are so close together that we have numerical problems. Since they're so close, we can just ignore this case anyway, except possibly
			% display that numerical issues may have occurred.
			disp(['There may be numerical issues in the challenging process at particle ',num2str(i),'.']);
		end
	end
	
	% Finally we set up the maximum distribution, given all the particles.
	for i = 1:np
		pMax(round+1,:) = pMax(round+1,:) + weights(i)*1/sqrt(det(2*pi*h^2))*exp(-1/2*(Xs - particles(i)).^2/h^2);
	end
	pMax(round+1,:) = pMax(round+1,:)/sum(weights);
end

% Next, we calculate the limit distribution of the particles.
P = zeros(ns,ns);
for i = 1:ns
	for j = 1:ns
		mut = mPost(i) - mPost(j);
		Sigmat = SPost(i,i) + SPost(j,j) - SPost(i,j) - SPost(j,i);
		P(i,j) = erf(mut/sqrt(2*Sigmat))/2 + 1/2;
	end
	P(i,i) = 1/2;
end

% We calculate the comparison matrix and use it to find the limit distribution of the particles.
mat = diag(diag(ones(ns,ns)*P)) - P;
outcome = zeros(ns,1);
mat(end,:) = ones(1,ns); % We set the bottom row equal to ones.
outcome(end) = 1; % We set the bottom element of the outcome equal to one.
limitDist = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
limitDist = limitDist/dx; % We turn the result into a PDF.

% We plot the results.
figure(17);
clf(17);
hold on;
grid on;
xlabel('Input');
ylabel('Particle distribution');
for i = 1:nr+1
	particleDistribution = plot(Xs, pMax(i,:), '-', 'Color', 0.8*(1 - i/(nr+1))*[1,1,1]);
end
axis([xMin,xMax,0,2.5]);

% We also add the true maximum distribution and the limit distribution to the plot.
limitDistribution = plot(Xs, limitDist, '-', 'Color', red, 'LineWidth', 1);
load('TrueMaximumDistribution');
trueDistribution = plot(Xs, maxDist, '-', 'Color', blue);
legend([trueDistribution, limitDistribution, particleDistribution], 'True distribution', 'Limit distribution', 'Particle distribution', 'Location', 'NorthWest');
axis([xMin,xMax,0,2.5]);
if exportFigs ~= 0
	export_fig('ImprovedParticleDistribution.png','-transparent');
end

% We set up Figure 6.9. based on the results of the particle method we just ran.
disp('Creating Figure 6.9.');

% We set up the probability distribution for the maximum value, based on the particles.
pfMax = zeros(1,ns);
for i = 1:np
	pfMax = pfMax + weights(i)*1/sqrt(det(2*pi*hf^2))*exp(-1/2*(fm - values(i)).^2/hf^2);
end
pfMax = pfMax/sum(weights);

% We set up the plot for the maximum value distribution.
figure(18);
clf(18);
hold on;
grid on;
xlabel('Output');
ylabel('Maximum value distribution');
particleDistribution = plot(fm, pfMax, '-', 'Color', black);

% We add the true maximum value distribution to it, found using brute force methods.
load('TrueMaximumValueDistribution');
trueDistribution = plot(fm, maxValDist, '-', 'Color', blue);
legend([trueDistribution, particleDistribution], 'True distribution', 'Particle distribution', 'Location', 'NorthWest');
if exportFigs ~= 0
	export_fig('MaximumValueDistribution.png','-transparent');
end
