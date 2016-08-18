% This file contains all the scripts for Chapter 3 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter3 from the Matlab command.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.
addpath('../Tools'); % This is for a few useful add-on functions, like the logdet function.

%% Figure 3.1.
disp('Creating Figure 3.1.');
% We set up hyperparameters for the GP which we will be sampling data from.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.2; % This is the output noise scale.
xMin = 0; % This is the minimum value for x.
xMax = 10; % This is the maximum value for x.
nm = 20; % This is the number of measurements we will do.
rng(8, 'twister'); % We fix the random number generator to a state which I know works well, so we always get the same useful outcome.

% We take nm random input points and, according to the GP distribution, we randomly sample output values from it.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
diff = repmat(Xm,nm,1) - repmat(Xm',1,nm); % This is matrix containing differences between input points.
Kmm = lf^2*exp(-1/2*diff.^2/lf^2); % This is the covariance matrix. It contains the covariances of each combination of points.
fm = mvnrnd(zeros(nm,1),Kmm)'; % We set up exact function values from a sample function.
fmh = fm + sfh*randn(nm,1); % We distort these exact function values by random noise with the right covariance.

% Next, we set up a set of trial input points and calculate the resulting posterior distribution of the trial function values.
Xs = xMin:0.01:xMax; % These are the trial points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
logp = -nm/2*log(2*pi) - 1/2*logdet(Kmm + Sfm) - 1/2*(fmh - mm)'/(Kmm + Sfm)*(fmh - mm); % This is the log-likelihood.
disp(['For the first hyperparameter effect plot, the log-likelihood is ',num2str(logp),'.']);

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
axis([xMin,xMax,-3,2]);
if exportFigs ~= 0
	export_fig('HyperparameterEffect1.png','-transparent');
end

% Now we adjust the hyperparameters and do it all again.
lf = 1; % This is the output length scale.
lx = 0.5; % This is the input length scale.
sfh = 0.04; % This is the output noise scale.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
logp = -nm/2*log(2*pi) - 1/2*logdet(Kmm + Sfm) - 1/2*(fmh - mm)'/(Kmm + Sfm)*(fmh - mm); % This is the log-likelihood.
disp(['For the second hyperparameter effect plot, the log-likelihood is ',num2str(logp),'.']);

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
axis([xMin,xMax,-3,2]);
if exportFigs ~= 0
	export_fig('HyperparameterEffect2.png','-transparent');
end

% Now we adjust the hyperparameters and do it all again.
lf = 1; % This is the output length scale.
lx = 2; % This is the input length scale.
sfh = 1; % This is the output noise scale.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
logp = -nm/2*log(2*pi) - 1/2*logdet(Kmm + Sfm) - 1/2*(fmh - mm)'/(Kmm + Sfm)*(fmh - mm); % This is the log-likelihood.
disp(['For the third hyperparameter effect plot, the log-likelihood is ',num2str(logp),'.']);

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
axis([xMin,xMax,-3,2]);
if exportFigs ~= 0
	export_fig('HyperparameterEffect3.png','-transparent');
end

%% Figure 3.2.
disp('Creating Figure 3.2.');
% We set up hyperparameters for the GP which we will be sampling data from.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.2; % This is the output noise scale.
xMin = 0; % This is the minimum value for x.
xMax = 10; % This is the maximum value for x.
nm = 20; % This is the number of measurements we will do.
rng(8, 'twister'); % We fix the random number generator to a state which I know works well, so we always get the same useful outcome.

% We take nm random input points and, according to the GP distribution, we randomly sample output values from it.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
diff = repmat(Xm,nm,1) - repmat(Xm',1,nm); % This is matrix containing differences between input points.
Kmm = lf^2*exp(-1/2*diff.^2/lf^2); % This is the covariance matrix. It contains the covariances of each combination of points.
fm = mvnrnd(zeros(nm,1),Kmm)'; % We set up exact function values from a sample function.
fmh = fm + sfh*randn(nm,1); % We distort these exact function values by random noise with the right covariance.

% Next, we set hyperparameters to blatantly incorrect values, so we can tune them.
lf = 5;
lx = 0.2;
sfh = 2;
hyp = [lx^2;lf^2;sfh^2]; % This is an array of hyperparameters which will be tuned.

% We set things up for the gradient ascent algorithm.
numSteps = 100; % How many gradient ascent steps shall we take?
stepSize = 1; % This is the initial step size. In the non-dimensionalized gradient ascent algorithm we use below, this can be seen as a length scale of the optimized parameter, in this case the log-likelihood.
stepSizeFactor = 2; % This is the factor by which we will decrease the step size in case it is too big.
maxReductions = 100; % This is the maximum number of times in a row which we can reduce the step size. If we'd need this many reductions, something is obviously wrong.
clear logp; % We make sure that logp is not defined. Whether it is defined is used in the script to check if it's the first run of the algorithm.
newHypDeriv = zeros(3,1); % We already create a storage for the new hyperparameter derivative array. We'll need this soon.

% We set up a figure to plot results in.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');

% Now we can start iterating
for i = 1:numSteps
	% We try to improve the parameters, all the while checking the step size.
	for j = 1:maxReductions
		% We check if we haven't accidentally been decreasing the step size too much.
		if j == maxReductions
			disp('Error: something is wrong with the step size in the hyperparameter optimization scheme.');
		end
		% We calculate new hyperparameters. Or at least, candidates. We still check them.
		if ~exist('logp','var') % If no logp is defined, this is the first time we are looping. In this case, with no derivative data known yet either, we keep the hyperparameters the same.
			newHyp = hyp;
		else
			newHyp = hyp.*(1 + stepSize*hyp.*hypDeriv); % We apply a non-dimensional update of the hyperparameters. This only works when the parameters are always positive.
		end
		% Now we check the new hyperparameters. If they are good, we will implement them.
		if min(newHyp > 0) % The parameters have to remain positive. If they are not, something is wrong. To be precise, the step size is too big.
			% We partly implement the new hyperparameters and check the new value of logp.
			lx = sqrt(newHyp(1));
			lf = sqrt(newHyp(2));
			sfh = sqrt(newHyp(3));
			Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
			diff = repmat(Xm,nm,1) - repmat(Xm',1,nm); % This is a matrix with element [i,j] equal to x_j - x_i.
			Kmm = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
			P = Kmm + Sfm;
 			mb = (ones(nm,1)'/P*fmh)/(ones(nm,1)'/P*ones(nm,1)); % This is the (constant) mean function m(x) = \bar{m}. You can get rid of this line if you don't want to tune \bar{m}.
			newLogp = -nm/2*log(2*pi) - 1/2*logdet(P) - 1/2*(fmh - mb)'/P*(fmh - mb);
			% If this is the first time we are in this loop, or if the new logp is better than the old one, we fully implement the new hyperparameters and recalculate the derivative.
			if ~exist('logp','var') || newLogp >= logp
				% We calculate the new hyperparameter derivative.
				alpha = P\(fmh - mb);
				R = alpha*alpha' - inv(P);
				newHypDeriv(3) = 1/2*trace(R);
				newHypDeriv(2) = 1/(2*lf^2)*trace(R*Kmm);
				newHypDeriv(1) = 1/(4*lx^4)*trace(R*(Kmm.*(diff.^2)));
				% If this is not the first time we run this, we also update the step size, based on how much the (normalized) derivative direction has changed. If the derivative is still in the
				% same direction as earlier, we take a bigger step size. If the derivative is in the opposite direction, we take a smaller step size. And if the derivative is perpendicular to
				% what is used to be, then the step size was perfect and we keep it. For this scheme, we use the dot product.
				if exist('logp','var')
					directionConsistency = ((hypDeriv.*newHyp)'*(newHypDeriv.*newHyp))/norm(hypDeriv.*newHyp)/norm(newHypDeriv.*newHyp);
					stepSize = stepSize*stepSizeFactor^directionConsistency;
				end
				break; % We exit the step-size-reduction loop.
			end
		end
		% If we reach this, it means the hyperparameters we tried were not suitable. In this case, we should reduce the step size and try again. If the step size is small enough, there will
		% always be an improvement of the hyperparameters. (Unless they are fully perfect, which never really occurs.)
		stepSize = stepSize/stepSizeFactor;
	end
	% We update the important parameters.
	hyp = newHyp;
	hypDeriv = newHypDeriv;
	logp = newLogp;
	
	% For the first few iterations, we plot the mean.
	numLines = 20;
	if i <= numLines
		% We use the tuned hyperparameters to make a plot.
		lx = sqrt(hyp(1));
		lf = sqrt(hyp(2));
		sfh = sqrt(hyp(3));
		Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
		Xs = xMin:0.01:xMax; % These are the trial points.
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
		mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
		SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
		sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

		% We set up the GP plot.
		if useColor == 0
			plot(Xs, mPost, 'b-', 'LineWidth', 1, 'Color', [1-i/numLines,1-i/numLines,1-i/numLines]); % We plot the mean line.
		else
			plot(Xs, mPost, 'b-', 'LineWidth', 1, 'Color', [1-i/numLines,1-i/numLines,1]); % We plot the mean line.
		end
	end
end

% We display the results.
disp('After hyperparameter tuning we have:');
disp(['l_x = ',num2str(sqrt(hyp(1)))]);
disp(['l_f = ',num2str(sqrt(hyp(2)))]);
disp(['s_f = ',num2str(sqrt(hyp(3)))]);
disp(['\bar{m} = ',num2str(mb)]);
disp(['log(p) = ',num2str(logp)]);

% We add the measurement points to the plots and export it.
if useColor == 0
	plot(Xm, fmh, 'ko'); % We plot the measurement points.
else
	plot(Xm, fmh, 'ro'); % We plot the measurement points.
end
axis([xMin,xMax,-3,2]);
if exportFigs ~= 0
	export_fig('HyperparameterTuning.png','-transparent');
end

%% Figure 3.3.
disp('Creating Figure 3.3.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.1; % This is the output noise scale.
xMin = -3; % This is the minimum value for x.
xMax = 3; % This is the maximum value for x.
Xs = xMin:0.01:xMax; % These are the trial points.

% Next, we will generate some measurement data.
rng(19, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
nm = 7;
Xm = xMin + rand(1,nm)*(xMax - xMin);

% We now set up the (squared exponential) covariance matrix and related terms.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Now we will adjust the covariance matrix, to make sure it is zero whenever two input points are in a different region.
inRegion1 = (X < -1)*1; % We multiply by 1 to make sure Matlab stores the result as a number and not as a logical value, which will give errors later on.
inRegion2 = (X >= -1).*(X < 1)*1;
inRegion3 = (X >= 1)*1;
inSameRegion = (inRegion1'*inRegion1 + inRegion2'*inRegion2 + inRegion3'*inRegion3);
K = K.*inSameRegion;
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);

% Next, we apply GP regression. Although we don't use any measurements yet, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(3);
clf(3);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([xMin,xMax,-2.5*lf,2.5*lf]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(110, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PiecewiseContinuousGP.png','-transparent');
end

% Next, we generate measurement values and apply GP regression.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
fmh = mm + chol(Kmm + Sfm)'*randn(nm,1); % Note that we directly sample from the distribution of fmh, instead of first of fm and then adding noise.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(3);
clf(3);
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
axis([xMin,xMax,-2.5*lf,2.5*lf]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(14, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PiecewiseContinuousGPWithMeasurements.png','-transparent');
end

%% Figure 3.4.
disp('Creating Figure 3.4.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.1; % This is the output noise scale.
xMin = -6; % This is the minimum value for x.
xMax = 6; % This is the maximum value for x.
Xs = xMin:0.01:xMax; % These are the trial points.
p = 4; % This is the period.

% Next, we will generate some measurement data.
rng(9, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
nm = 7;
Xm = xMin + rand(1,nm)*(xMax - xMin);

% We now set up the (squared exponential) covariance matrix and related terms.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sin(pi*diff/p).^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression. Although we don't use any measurements yet, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([xMin,xMax,-2.5*lf,2.5*lf]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(13, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicGP.png','-transparent');
end

% Next, we generate measurement values and apply GP regression.
rng(27, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
fmh = mm + chol(Kmm + Sfm)'*randn(nm,1); % Note that we directly sample from the distribution of fmh, instead of first of fm and then adding noise.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(4);
clf(4);
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
axis([xMin,xMax,-2.5*lf,2.5*lf]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(14, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicGPWithMeasurements.png','-transparent');
end

%% Figure 3.5.
disp('Creating Figure 3.5.');
% We define data.
sfh = 1; % This is the output noise scale.
Kw = 1^2; % This is covariance of the weights.
kb = 4; % This is the standard deviation of the offset (or bias).
w = -0.8; % This is the real weight which we will plot data for.
b = 0; % This is the real offset which we will plot data for.
xMin = -5; % This is the minimum value for x.
xMax = 5; % This is the maximum value for x.
nm = 10; % This is the number of measurements we will do.
rng(6, 'twister'); % We fix the random number generator to a state which I know works well, so we always get the same useful outcome.

% We take nm random input points and set up measurement data for it.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
fm = Xm'*w;
fmh = fm + sfh*randn(nm,1); % We distort these exact function values by random noise with the right covariance.

% Next, we set up a set of trial input points and calculate the resulting posterior distribution of the trial function values.
Xs = xMin:0.01:xMax; % These are the trial points.
ns = size(Xs,2); % This is the number of trial points.
X = [Xm,Xs]; % We merge the measurement and trial points.
n = size(X,2); % This is the number of points.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
K = X'*Kw*X; % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We calculate the posterior distribution of w.
muw = (Xm/Sfm*Xm' + inv(Kw))\Xm/Sfm*fmh;
Sw = inv(Xm/Sfm*Xm' + inv(Kw));
disp(['Without offset, the posterior distribution of w has mean ',num2str(muw),' and standard deviation ',num2str(sqrt(Sw)),'.']);

% We set up the GP plot.
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
axis([xMin,xMax,-6,6]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(4, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('LinearGP1.png','-transparent');
end

% We now add kb^2 to the covariance vector, to incorporate an offset. We also add the offset b from all measurement to actually create an offset.
Kw = [Kw,0;0,kb^2];
Xs = [Xs;ones(1,ns)];
Xm = [Xm;ones(1,nm)];
X = [Xm,Xs]; % We merge the measurement and trial points.
K = X'*Kw*X; % This is the covariance matrix. It contains the covariances of each combination of points.
fmh = fmh + b;
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We calculate the posterior distribution of w.
muw = (Xm/Sfm*Xm' + inv(Kw))\Xm/Sfm*fmh;
Sw = inv(Xm/Sfm*Xm' + inv(Kw));
disp(['With offset, the posterior distribution of w has mean ',num2str(muw(1)),' and standard deviation ',num2str(sqrt(Sw(1,1))),'.']);
disp(['With offset, the posterior distribution of b has mean ',num2str(muw(2)),' and standard deviation ',num2str(sqrt(Sw(2,2))),'.']);

% We set up the GP plot.
figure(5);
clf(5);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs(1,:), fliplr(Xs(1,:))],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs(1,:), fliplr(Xs(1,:))],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs(1,:), mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm(1,:), fmh, 'ko'); % We plot the measurement points.
else
	patch([Xs(1,:), fliplr(Xs(1,:))],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs(1,:), fliplr(Xs(1,:))],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs(1,:), mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
	plot(Xm(1,:), fmh, 'ro'); % We plot the measurement points.
end
axis([xMin,xMax,-6+b,6+b]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(4, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs(1,:), sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs(1,:), sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs(1,:), sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('LinearGP2.png','-transparent');
end

%% Figure 3.6.
disp('Creating Figure 3.6.');
% We define data.
sfh = 1; % This is the output noise scale.
mu = 1; % This is the mean of the sinusoid.
A = 2; % This is the amplitude.
phase = pi/4; % This is the phase.
Kw = diag([4^2,4^2,4^2]); % These are the standard deviations for the parameters which we will estimate. (Note that we do not directly estimate the amplitude and the phase.)
xMin = -1; % This is the minimum value for x.
xMax = 1; % This is the maximum value for x.
nm = 10; % This is the number of measurements we will do.
rng(3, 'twister'); % We fix the random number generator to a state which I know works well, so we always get the same useful outcome.

% We take nm random input points and set up measurement data for it.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the measurement points.
fm = mu + A*sin(2*pi*Xm - phase)'; % We calculate the exact function values.
fmh = fm + sfh*randn(nm,1); % We distort these exact function values by random noise with the right covariance.

% Next, we set up a set of trial input points and calculate the resulting posterior distribution of the trial function values.
Xs = xMin:0.01:xMax; % These are the trial points.
ns = size(Xs,2); % This is the number of trial points.
fs = mu + A*sin(2*pi*Xs - phase)'; % We calculate the exact function values for the trial points.
Phis = [sin(2*pi*Xs);cos(2*pi*Xs);ones(1,ns)]; % This is the set of feature vectors for all trial points.
Phim = [sin(2*pi*Xm);cos(2*pi*Xm);ones(1,nm)]; % This is the set of feature vectors for all measurement points.
Phi = [Phim,Phis]; % We merge the measurement and trial points.
n = size(Phi,2); % This is the number of points.
Sfm = sfh^2*eye(nm); % This is the noise covariance matrix.
K = Phi'*Kw*Phi; % This is the covariance matrix. It contains the covariances of each combination of points.
Kmm = K(1:nm,1:nm);
Kms = K(1:nm,nm+1:end);
Ksm = Kms';
Kss = K(nm+1:end,nm+1:end);
mm = zeros(nm,1); % This is the mean vector m(Xm). We assume a zero mean function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We calculate the posterior distribution of w.
muw = (Phim/Sfm*Phim' + inv(Kw))\Phim/Sfm*fmh;
Sw = inv(Phim/Sfm*Phim' + inv(Kw));
disp(['The weights are as follows.']);
disp(['Weight 1 has mean ',num2str(muw(1)),' and standard deviation ',num2str(sqrt(Sw(1,1))),'.']);
disp(['Weight 2 has mean ',num2str(muw(2)),' and standard deviation ',num2str(sqrt(Sw(2,2))),'.']);
disp(['Weight 3 has mean ',num2str(muw(3)),' and standard deviation ',num2str(sqrt(Sw(3,3))),'.']);

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
axis([xMin,xMax,mu-1.5*A,mu+1.5*A]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('FeatureFunctionGP.png','-transparent');
end

%% Figure 3.7.
disp('Creating Figure 3.7.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.1; % This is the output noise scale.
Kw = 1/4^2; % This is the covariance matrix of the weights.
Xs = -6:0.01:6; % These are the trial points.
p = 4; % This is the period.

% We now set up the covariance matrix (and related terms) for a periodic plus a linear function.
ns = size(Xs,2); % This is the number of trial points.
diff = repmat(Xs,ns,1) - repmat(Xs',1,ns); % This is matrix containing differences between input points.
Kss = lf^2*exp(-1/2*sin(pi*diff/p).^2/lx^2) + Xs'*Kw*Xs; % This is the covariance matrix. The first part is for the periodic function. The second part is for the linear function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression. Although we don't have any measurements, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
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
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([min(Xs),max(Xs),-3,3]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(34, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicPlusLinearGP.png','-transparent');
end

% Next we set up the covariance matrix for the periodic plus smooth covariance function.
lx1 = 1/2; % This is the small input length scale for the periodic function.
lf1 = 1/2; % This is the small output length scale for the periodic function.
lx2 = 5; % This is the large input length scale for the smooth function.
lf2 = 2; % This is the large output length scale for the smooth function.
p = 1; % This is the new period.
Kss = lf1^2*exp(-1/2*sin(pi*diff/p).^2/lx1^2) + lf2^2*exp(-1/2*diff.^2/lx2^2); % This is the piecewise linear covariance matrix plus the squared exponential covariance matrix.

% Next, we apply GP regression. Although we don't have any measurements, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(8);
clf(8);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([min(Xs),max(Xs),-5,5]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(33, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicPlusSmoothGP.png','-transparent');
end

%% Figure 3.8.
disp('Creating Figure 3.8.');
% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfh = 0.1; % This is the output noise scale.
Kw = 1/4^2; % This is the covariance matrix of the weights.
Xs = -6:0.01:6; % These are the trial points.
p = 4; % This is the period.

% We now set up the covariance matrix (and related terms) for a periodic plus a linear function.
ns = size(Xs,2); % This is the number of trial points.
diff = repmat(Xs,ns,1) - repmat(Xs',1,ns); % This is matrix containing differences between input points.
Kss = lf^2*exp(-1/2*sin(pi*diff/p).^2/lx^2).*(Xs'*Kw*Xs); % This is the covariance matrix. The first part is for the periodic function. The second part is for the linear function.
ms = zeros(ns,1); % This is the mean vector m(Xs). We assume a zero mean function.

% Next, we apply GP regression. Although we don't have any measurements, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(8);
clf(8);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([min(Xs),max(Xs),-3,3]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(34, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicTimesLinearGP.png','-transparent');
end

% Next we set up the covariance matrix for the periodic plus smooth covariance function.
lx1 = 1/2; % This is the small input length scale for the periodic function.
lf1 = 1/2; % This is the small output length scale for the periodic function.
lx2 = 5; % This is the large input length scale for the smooth function.
lf2 = 2; % This is the large output length scale for the smooth function.
p = 1; % This is the new period.
Kss = (lf1^2*exp(-1/2*sin(pi*diff/p).^2/lx1^2)).*(lf2^2*exp(-1/2*diff.^2/lx2^2)); % This is the piecewise linear covariance matrix plus the squared exponential covariance matrix.

% Next, we apply GP regression. Although we don't have any measurements, so that's easy.
mPost = ms; % This is the posterior mean vector.
SPost = Kss; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We set up the GP plot.
figure(8);
clf(8);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
if useColor == 0
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [1,1,1]*0.9, 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [1,1,1]*0.8, 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'k-', 'LineWidth', 1); % We plot the mean line.
else
	patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', [0.9,0.9,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', [0.8,0.8,1], 'EdgeColor', 'none'); % This is the grey area in the plot.
	set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
	plot(Xs, mPost, 'b-', 'LineWidth', 1); % We plot the mean line.
end
axis([min(Xs),max(Xs),-3,3]);

% Next, we generate samples from the posterior distribution of the trial points.
rng(192, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
eps = 1e-10; % We use a small number to add to the diagonal (so to the eigenvalues) of the matrix, to make sure Matlab doesn't give any numerical issues with finding the Cholesky decomposition.
SPostCholesky = chol(SPost + eps*eye(ns)); % We use the Cholesky decomposition to sample from the posterior Gaussian distribution.
sample1 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample1, 'k--');
sample2 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
plot(Xs, sample2, 'k-.');
% sample3 = mPost + SPostCholesky'*randn(ns,1); % We generate a sample from the Gaussian distribution with mean mPost and with covariance matrix SPost.
% plot(Xs, sample3, 'k:');
rng('shuffle'); % And we unfix Matlab's random number generator again.

% And we export the graph too.
if exportFigs ~= 0
	export_fig('PeriodicTimesSmoothGP.png','-transparent');
end