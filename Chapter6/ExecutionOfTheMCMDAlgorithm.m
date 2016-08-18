% This file contains the experiment on executing the MCMD algorithm. With the predefined parameters, running this script should take up to ten seconds. Of course if more measurements, particles
% or challenge rounds are added, the runtime will increase.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

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

% We generate measurements for the example GP which we will use.
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
greyArea = patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
meanLine = plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
sample1Plot = plot(Xs, sample1, '-', 'Color', green); % We draw the three different samples.
sample2Plot = plot(Xs, sample2, '-', 'Color', yellow);
sample3Plot = plot(Xs, sample3, '-', 'Color', brown);
plot(Xs(i1), sample1(i1), 'x', 'Color', green); % We use a cross to indicate the maximums.
plot(Xs(i2), sample2(i2), 'x', 'Color', yellow);
plot(Xs(i3), sample3(i3), 'x', 'Color', brown);
% plot(Xs, fs, '-', 'LineWidth', 1, 'Color', black); % This plots the true function which we are approximating.
measurementPoints = plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
axis([xMin,xMax,-2,1.5]);
legend([meanLine,greyArea,measurementPoints],'GP mean','GP 95% region','Measurements','Location','SouthEast');
if exportFigs ~= 0
	export_fig('GPExampleWithSamples.png','-transparent');
end

% Next, we will find the true maximum distribution through brute force methods.

% We generate a lot of samples and find their maximums. Then we use this to find the maximum distribution.
nSamples = 1e5; % This is the number of sample functions we will examine.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
maxCounter = zeros(1,ns); % We will use this array to count how many times a specific point is the maximum.
for i = 1:nSamples
	sample = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian process.
	[val,ind] = max(sample); % We find out where the maximum occurs. That is, at which of the ns trial points.
	maxCounter(ind) = maxCounter(ind) + 1; % We increment the counter for this trial point.
end
maxDist = maxCounter/nSamples/dx; % This is the maximum distribution. It is a probability density function, meaning its integral adds up to one.

% We also calculate the limit distribution of the particles.

% First we set up the P matrix, calculating probabilities that one point is larger than another. We then calculate the comparison matrix and use it to find the limit distribution of the particles.
P = zeros(ns,ns);
for i = 1:ns
	for j = 1:ns
		mut = mPost(i) - mPost(j);
		Sigmat = SPost(i,i) + SPost(j,j) - SPost(i,j) - SPost(j,i);
		P(i,j) = erf(mut/sqrt(2*Sigmat))/2 + 1/2;
	end
	P(i,i) = 1/2;
end
mat = diag(diag(ones(ns,ns)*P)) - P;
outcome = zeros(ns,1);
mat(end,:) = ones(1,ns); % We set the bottom row equal to ones.
outcome(end) = 1; % We set the bottom element of the outcome equal to one.
limitDist = mat\outcome; % These are the probabilities that each point is larger than any of the other points, according to the particle method.
limitDist = limitDist/dx; % We turn the result into a PDF.

% The next step is to run the MCMD algorithm.

% We define parameters.
nr = 10; % We define the number of rounds we apply.
np = 1e4; % We define the number of particles used.
alpha = 0.5; % This is the part of the time we sample a challenger from the current belief of the maximum distribution. A high value of alpha (near 1) speeds up convergence, but distorts the results slightly. For larger problems it is wise to start with a high value of alpha but decrease it towards zero as the algorithm converges.
h = 0.04; % This is the length scale of the Gaussian kernel we will use in the kernel density estimation process. It has been tuned to be the smallest number to still give smooth plots.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it gives the plot shown in the thesis.

% We initialize the particles.
particles = xMin + (xMax - xMin)*rand(1,np); % We set up particles in random locations.
weights = ones(np,1); % We initialize the weights.

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

% We plot the results.
figure(2);
clf(2);
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
trueDistribution = plot(Xs, maxDist, '-', 'Color', blue);
legend([trueDistribution, limitDistribution, particleDistribution], 'True distribution', 'Limit distribution', 'Particle distribution', 'Location', 'NorthWest');
axis([xMin,xMax,0,2.5]);
if exportFigs ~= 0
	export_fig('GPExampleMaximumDistribution.png','-transparent');
end