% This file compares the NIGP regression and the SONIG regression algorithm using a plot of a small number of measurement points.

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

% We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
rng(6, 'twister');

% We define the range of the plot we will make.
xMin = -5; % What is the minimum x value?
xMax = -xMin; % What is the maximum x value?

% We define numbers of points and set up the corresponding point spaces.
nm = 30; % This is the number of available measurement points.
ns = 101; % This is the number of plot (trial) points.
nu = 11; % The number of inducing input points.
Xs = linspace(xMin,xMax,ns); % These are the plot points.
Xu = linspace(xMin,xMax,nu); % These are the inducing input points.

% We define some settings for the noise and the GP.
sfm = 0.1; % This is the noise standard deviation on the function output.
sxm = 0.4; % This is the noise standard deviation on the function input.
lf = 1; % This is the length scale of the output.
lx = 1; % This is the length scale for the input. So it's the square root of Lambda.
Lambda = lx^2;

% We set up the input points.
Xm = xMin + rand(1,nm)*(xMax - xMin); % These are the real measurement input points without noise.
Xmh = Xm + sxm*randn(1,nm); % These are the measured input points.

% We calculate covariance matrices.
input = [Xu,Xmh,Xs,Xm];
diff = repmat(input,[size(input,2),1]) - repmat(input',[1,size(input,2)]);
K = lf^2*exp(-1/2*diff.^2/Lambda);
KDivided = mat2cell(K,[nu,nm,ns,nm],[nu,nm,ns,nm]);
Kuu = KDivided{1,1};
Kum = KDivided{1,2};
Kus = KDivided{1,3};
Kur = KDivided{1,4};
Kmu = KDivided{2,1};
Kmm = KDivided{2,2};
Kms = KDivided{2,3};
Kmr = KDivided{2,4};
Ksu = KDivided{3,1};
Ksm = KDivided{3,2};
Kss = KDivided{3,3};
Ksr = KDivided{3,4};
Kru = KDivided{4,1};
Krm = KDivided{4,2};
Krs = KDivided{4,3};
Krr = KDivided{4,4};

% To generate a random sample with covariance matrix K, we first have to find the Cholesky decomposition of K. That's what we do here.
epsilon = 0.0000001; % We add some very small noise to prevent K from being singular.
L = chol([Krr,Krs;Ksr,Kss] + epsilon*eye(nm+ns))'; % We take the Cholesky decomposition to be able to generate a sample with a distribution according to the right covariance matrix. (Yes, we could also use the mvnrnd function, but that one gives errors more often than the Cholesky function.)
sample = L*randn(nm+ns,1);

% We create the measurements.
fm = sample(1:nm)'; % These are the real function measurements, done at the real measurement input points, without any noise.
fmh = fm + sfm*randn(1,nm); % We add noise to the function measurements, to get the noisy measurements.
fs = sample(nm+1:nm+ns)'; % This is the function value of the function we want to approximate at the plot points.

% We make a plot of the function which we want to approximate, including the real measurements and the noisy measurements.
figure(1);
clf(1);
hold on;
grid on;
plot(Xs, fs, 'b-');
plot(Xm(1:nm), fm(1:nm), 'g+');
plot(Xmh(1:nm), fmh(1:nm), 'ro');
xlabel('Input');
ylabel('Output');
legend('Real function','Noiseless measurements','Noisy measurements','Location','SouthEast');

% The next step is to train the NIGP algorithm. We start doing that now.
seard = log([lx;lf;sfm]); % We give the NIGP algorithm the true hyperparameters as starting point for its tuning. It's slightly cheating, but NIGP is likely to find the same hyperparameters with similar initializations, so this just speeds things up a little bit.
lsipn = log(sxm);
evalc('[model, nigp] = trainNIGP(permute(Xmh,[2,1]),permute(fmh,[2,1]),-500,1,seard,lsipn);'); % We apply the NIGP training algorithm. We put this in an evalc function to suppress the output made by the NIGP algorithm.

% We extract the derived hyperparameters from the NIGP results.
lx = exp(model.seard(1,1));
lf = exp(model.seard(2,1));
sfm = exp(model.seard(3,1));
sxm = exp(model.lsipn);
Lambda = lx^2;
Sx = sxm^2;
disp(['Hyperparameters found. lx: ',num2str(lx),', sx: ',num2str(sxm),', ly: ',num2str(lf),', sy: ',num2str(sfm),'.']);

% We recalculate covariance matrices for the new hyperparameters.
input = [Xu,Xmh,Xs,Xm];
diff = repmat(input,[size(input,2),1]) - repmat(input',[1,size(input,2)]);
K = lf^2*exp(-1/2*diff.^2/Lambda);
KDivided = mat2cell(K,[nu,nm,ns,nm],[nu,nm,ns,nm]);
Kuu = KDivided{1,1};
Kum = KDivided{1,2};
Kus = KDivided{1,3};
Kur = KDivided{1,4};
Kmu = KDivided{2,1};
Kmm = KDivided{2,2};
Kms = KDivided{2,3};
Kmr = KDivided{2,4};
Ksu = KDivided{3,1};
Ksm = KDivided{3,2};
Kss = KDivided{3,3};
Ksr = KDivided{3,4};
Kru = KDivided{4,1};
Krm = KDivided{4,2};
Krs = KDivided{4,3};
Krr = KDivided{4,4};

% We make the NIGP prediction for the test points.
musNIGP = Ksm(:,1:nm)/(Kmm(1:nm,1:nm) + sfm^2*eye(nm) + diag(model.dipK(1:nm)))*fmh(1:nm)'; % This is the mean at the plot points.
SssNIGP = Kss - Ksm(:,1:nm)/(Kmm(1:nm,1:nm) + sfm^2*eye(nm) + diag(model.dipK(1:nm)))*Kms(1:nm,:); % This is the covariance matrix at the plot points.
stdsNIGP = sqrt(diag(SssNIGP)); % This is the standard deviation at the plot points.

% We plot the results.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[musNIGP-2*stdsNIGP; flipud(musNIGP+2*stdsNIGP)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
hStd = patch([Xs, fliplr(Xs)],[musNIGP-stdsNIGP; flipud(musNIGP+stdsNIGP)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
hMean = plot(Xs, musNIGP, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
hMeasurements = plot(Xmh(1:nm), fmh(1:nm), 'o', 'Color', red);
hFunction = plot(Xs, fs, '-', 'Color', black);
xlabel('Input');
ylabel('Output');
% legend([hFunction,hMeasurements,hMean,hStd],'Original function','Measurements','GP prediction mean','GP 95% certainty region','Location','SouthEast');
axis([xMin,xMax,-2,3]);
if exportFigs ~= 0
	export_fig('NIGPPredictionExample.png','-transparent');
end

% We examine the results.
MSE = mean((musNIGP' - fs).^2);
meanVariance = mean(stdsNIGP.^2);
disp(['For NIGP the MSE is ',num2str(MSE),', the mean variance is ',num2str(meanVariance),' and the ratio between these is ',num2str(MSE/meanVariance),'.']);

% Next, we set up the SONIG algorithm to make a similar kind of prediction.
% We start by setting up a SONIG object which we can apply GP regression on.
hyp = NIGPModelToHyperparameters(model);
sonig = createSONIG(hyp);
sonig = addInducingInputPoint(sonig, Xu);

% We implement the measurements one by one.
for i = 1:nm
	inputDist = createDistribution(Xmh(:,i), hyp.sx^2); % This is the prior distribution of the input point.
	outputDist = createDistribution(fmh(:,i), hyp.sy^2); % This is the prior distribution of the output point.
	[sonig, inputPost, outputPost] = implementMeasurement(sonig, inputDist, outputDist); % We implement the measurement into the SONIG object.
end

% We predict the plot points and make a plot out of it.
[musSONIG, SssSONIG, stdsSONIG] = makeSonigPrediction(sonig, Xs); % Here we make the prediction.
figure(3);
clf(3);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[musSONIG-2*stdsSONIG; flipud(musSONIG+2*stdsSONIG)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[musSONIG-stdsSONIG; flipud(musSONIG+stdsSONIG)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, musSONIG, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xmh(1:nm), fmh(1:nm), 'o', 'Color', red);
plot(Xs, fs, '-', 'Color', black);
errorbar(sonig.Xu, sonig.fu{1}.mean, 2*sqrt(diag(sonig.fu{1}.cov)), '*', 'Color', yellow); % We plot the inducing input points.
axis([xMin,xMax,-2,3]);
if exportFigs ~= 0
	export_fig('SONIGPredictionExample.png','-transparent');
end

% We examine the results.
MSE = mean((musSONIG' - fs).^2);
meanVariance = mean(stdsSONIG.^2);
disp(['For SONIG the MSE is ',num2str(MSE),', the mean variance is ',num2str(meanVariance),' and the ratio between these is ',num2str(MSE/meanVariance),'.']);