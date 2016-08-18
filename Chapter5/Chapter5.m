% This file contains all the scripts for Chapter 5 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter5 from the Matlab command.

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
else
	red = [0.8 0 0];
	green = [0 0.4 0];
	blue = [0 0 0.8];
	yellow = [0.6 0.6 0];
	grey = [0.8 0.8 1];
end

%% Figure 5.1.
disp('Creating Figure 5.1.');

% We define data.
lf = 1; % This is the output length scale.
lx = 0.5; % This is the input length scale.
sfm = 0.1; % This is the output noise scale.
minX = -1; % This is the lower bound of the input space.
maxX = 1; % This is the upper bound of the input space.
minY = 0; % This is the lower bound of the output for the plot.
maxY = 3; % This is the upper bound of the output for the plot.
Xs = minX:0.01:maxX; % These are the trial points.
ns = length(Xs); % This is the number of trial points.
nm = 6; % How many measurements do we use?
xh = 0; % What is the mean of the trial input point we use?
Sx = 0.3^2; % What is the variance of the trial input point?
pxh = 1/sqrt(det(2*pi*Sx))*exp(-1/2*(Xs - xh).^2/Sx); % This is the probability density function of the trial input point.

% We set up measurement input points. These are manually chosen to get a decent graph.
Xm = linspace(-0.9,0.9,nm);
fmh = [1.7;1.3;1.5;2.6;2.2;1.5];

% We set up the covariance matrices.
X = [Xm,Xs];
n = size(X,2);
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nm,ns],[nm,ns]);
Kmm = KDivided{1,1};
Kms = KDivided{1,2};
Ksm = KDivided{2,1};
Kss = KDivided{2,2};
mm = zeros(nm,1); % This is the prior mean vector of the measurement points.
ms = zeros(ns,1); % This is the prior mean vector of the trial points.
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.

% We apply GP regression to the measurements.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results. That is, we both plot the resulting GP as well as the probability density of the trial input we will put into this.
figure(1);
clf(1);
hold on;
grid on;
xlabel('Input');
ylabel(['\color[rgb]{',num2str(green(1)),',',num2str(green(2)),',',num2str(green(3)),'}Input probability density\color{black} / \color[rgb]{',num2str(blue(1)),',',num2str(blue(2)),',',num2str(blue(3)),'}Gaussian process output']);
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
patch([minX,Xs,maxX], [0,pxh,0], 1, 'FaceColor', (green+2*white)/3, 'EdgeColor', 'none'); % We plot the input PDF.
axis([minX,maxX,minY,maxY]); % We make sure the axes are set up for the right data range.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey areas.
if exportFigs ~= 0
	export_fig('ExampleGPWithStochasticInput.png','-transparent');
end

% We now calculate the posterior distribution of the output.
ys = linspace(minY, maxY, ns); % We set up a plot index for the output.
pf = zeros(1,ns); % This is the array which we will eventually put the posterior probability density of f_* in.
dx = (maxX - minX)/(ns - 1); % This is the size of the steps in Xs, which we need to process the probability density of xh.
for i = 1:ns % We now walk through all the x-points, insert their value into the GP, weigh them according to the probability that this actually is the input, and add up the result.
	currX = Xs(i);
	currMean = mPost(i);
	currVar = SPost(i,i);
	pf = pf + (pxh(i)*dx)*1/sqrt(det(2*pi*currVar))*exp(-1/2*(ys - currMean).^2/currVar); % We add the PDF of the Gaussian distribution for f at this input point, and weigh it by the corresponding probability that this is the actual input point.
end

% We also calculate the approximated Gaussian distribution through moment matching.
Ksmbar = lf^2*sqrt(det(lx^2)/det(lx^2 + Sx))*exp(-1/2*(repmat(Xm,1,1) - repmat(xh',1,nm)).^2/(lx^2 + Sx)); % This is the covariance matrix between Xm and X* with uncertainty incorporated.
msbar = 0;
alpha = (Kmm + Sfm)\(fmh - mm);
mus = msbar + Ksmbar*alpha; % This is the posterior mean vector.
Q = eye(nm)/(Kmm + Sfm) - alpha*alpha';
Sss = lf^2 - mus^2; % We will calculate the posterior variance as this parameter. For that, we use a summation.
for i = 1:nm % There probably is a more computationally efficient way of calculating this in Matlab than with the overly slow for-loops, but I got a bit lazy here.
	for j = 1:nm
		Sss = Sss - lf^4*sqrt(det(lx^2)/det(lx^2 + 2*Sx))*Q(i,j)*exp(-1/2*(Xm(i) - Xm(j))^2/(2*lx^2))*exp(-1/2*((Xm(i) + Xm(j))/2 - xh)^2/(1/2*lx^2 + Sx));
	end
end
pfApprox = 1/sqrt(det(2*pi*Sss))*exp(-1/2*(ys - mus).^2/Sss); % This is the probability density function of the moment matched Gaussian distribution.

% We plot the result in a tilted graph, so it corresponds well with the previous plot.
figure(2);
clf(2);
hold on;
grid on;
patch([0,pf,0], [minY,ys,maxY], 1, 'FaceColor', (blue+3*white)/4, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(pfApprox, ys, '-', 'Color', blue, 'LineWidth', 1);
xlabel('Output probability density');
ylabel('Gaussian process output');
legend('True output probability density','Moment matching approximation','Location','SouthEast');
if exportFigs ~= 0
	export_fig('ResultingOutputPDF.png','-transparent');
end

%% Figure 5.2. and 5.3.
disp('Creating Figures 5.2. and 5.3.');

% We define data.
lf = 3; % This is the output length scale.
lx = 0.5; % This is the input length scale.
sfm = 0.4; % This is the output noise scale.
minX = -1; % This is the lower bound of the input space.
maxX = 1; % This is the upper bound of the input space.
minY = 0; % This is the lower bound of the output for the plot.
maxY = 8; % This is the upper bound of the output for the plot.
Xs = minX:0.01:maxX; % These are the trial points.
ns = length(Xs); % This is the number of trial points.
nm = 6; % How many measurements do we use?
xh = 0; % What is the mean of the measurement input point we use?
fh = 5; % What is the measured value that we have?
Sx = 0.3^2; % What is the variance of the measurement input point?
pxh = 1/sqrt(det(2*pi*Sx))*exp(-1/2*(Xs - xh).^2/Sx); % This is the probability density function of the true measurement input point.

% We set up measurement input points and inducing input points. These are manually chosen to get a decent graph.
Xm = linspace(-0.9,0.9,nm);
fmh = [3;5.1;6.6;6.9;5.7;3.9];
Xu = Xm;
nu = size(Xu,2);

% We set up the covariance matrices.
X = [Xm,Xu,Xs];
n = size(X,2);
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nm,nu,ns],[nm,nu,ns]);
Kmm = KDivided{1,1};
Kmu = KDivided{1,2};
Kms = KDivided{1,3};
Kum = KDivided{2,1};
Kuu = KDivided{2,2};
Kus = KDivided{2,3};
Ksm = KDivided{3,1};
Ksu = KDivided{3,2};
Kss = KDivided{3,3};
mm = zeros(nm,1); % This is the prior mean vector of the measurement points.
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.
ms = zeros(ns,1); % This is the prior mean vector of the trial points.
Sfm = sfm^2*eye(nm);

% We apply (offline) FITC regression to the measurements.
Lmm = diag(diag(Kmm - Kmu/Kuu*Kum));
Duu = Kuu + Kum/(Lmm + Sfm)*Kmu;
muu = mu + Kuu/Duu*Kum/(Lmm + Sfm)*(fmh - mm);
Suu = Kuu/Duu*Kuu;
su = sqrt(diag(Suu));

% Next, we use the inducing input point distribution to calculate the posterior distribution.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
SmPost = SPost + sfm^2*eye(ns); % Here we add noise to get the prior GP for new measurements.
smPost = sqrt(diag(SmPost));

% We plot the results. That is, we both plot the resulting GP as well as the prior distribution of the measurement we have found.
figure(3);
clf(3);
hold on;
grid on;
xlabel('Input');
ylabel(['\color[rgb]{',num2str(red(1)),',',num2str(red(2)),',',num2str(red(3)),'}Input probability density\color{black} / \color[rgb]{',num2str(blue(1)),',',num2str(blue(2)),',',num2str(blue(3)),'}Gaussian process output']);
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([minX,Xs,maxX], [0,pxh,0], 1, 'FaceColor', (red+2*white)/3, 'EdgeColor', 'none'); % We plot the input PDF.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu, muu, 2*su, '*', 'Color', yellow); % We plot the inducing input points.
errorbar(xh, fh, sfm, 'o', 'Color', red); % We plot the measurement point.
axis([minX,maxX,minY,maxY]); % We make sure the axes are set up for the right data range.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey areas.
if exportFigs ~= 0
	export_fig('StochasticMeasurementPrior.png','-transparent');
end

% We now calculate the posterior distribution of the measurement input point.
px = pxh.*1./sqrt(2*pi*smPost'.^2).*exp(-1/2*(mPost' - fh).^2./smPost'.^2); % We multiply the prior probability pxh by the confirmation of the measurement.
dx = (maxX - minX)/(ns - 1); % This is the size of the steps in Xs, which we need to process the probability density of xh.
px = px/(sum(px)*dx); % We normalize the result.

% We plot the results. That is, we both plot the GP with measurement noise incorporated in its predictions, as well as the posterior probability density of the measurement input point.
figure(4);
clf(4);
hold on;
grid on;
xlabel('Input');
ylabel(['\color[rgb]{',num2str(red(1)),',',num2str(red(2)),',',num2str(red(3)),'}Input probability density\color{black} / \color[rgb]{',num2str(blue(1)),',',num2str(blue(2)),',',num2str(blue(3)),'}Gaussian process output']);
patch([Xs, fliplr(Xs)],[mPost-2*smPost; flipud(mPost+2*smPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-smPost; flipud(mPost+smPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([minX,Xs,maxX], [0,px,0], 1, 'FaceColor', (red+2*white)/3, 'EdgeColor', 'none'); % We plot the input PDF.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu, muu, 2*su, '*', 'Color', yellow); % We plot the inducing input points.
plot(xh, fh, 'o', 'Color', red); % We plot the measurement point.
plot([minX,maxX], [fh,fh], '-', 'Color', red); % We plot the horizontal line through the measurement point.
axis([minX,maxX,minY,maxY]); % We make sure the axes are set up for the right data range.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey areas.
if exportFigs ~= 0
	export_fig('StochasticMeasurementPosterior.png','-transparent');
end

% We now define data for setting up the linearization.
xb = -0.4; % This is the linearization point.
diff = repmat(Xu,1,1) - repmat(xb',1,nu); % This is a difference matrix between the linearization point and the inducing input points.
Kbu = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance between the linearization point function value and the inducing function values. We use the subscript b to refer to \bar{x}, the linearization point.
Kub = Kbu';
Kbb = lf^2; % This is the prior variance of the linearization point itself.
dKbu = Kbu.*diff/lx^2; % This is the derivative of Kbu.
mub = 0 + Kbu/Kuu*(muu - mu);
dmub = 0 + dKbu/Kuu*(muu - mu);
Sbb = Kbb - Kbu/Kuu*(Kuu - Suu)/Kuu*Kub; % This is the variance of the output at the linearization point.
Shbb = Sbb + sfm^2; % This is the variance of the measured output (including measurement noise) at the linearization point.

% We now set up the linearized GP.
mPost = mub + dmub*(Xs - xb)';
SPost = ones(ns,ns)*Sbb;
sPost = sqrt(diag(SPost));
SmPost = SPost + sfm^2*eye(ns);
smPost = sqrt(diag(SmPost));

% We plot the result, similarly to what we did before at figure(3).
figure(5);
clf(5);
hold on;
grid on;
xlabel('Input');
ylabel(['\color[rgb]{',num2str(red(1)),',',num2str(red(2)),',',num2str(red(3)),'}Input probability density\color{black} / \color[rgb]{',num2str(blue(1)),',',num2str(blue(2)),',',num2str(blue(3)),'}Gaussian process output']);
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([minX,Xs,maxX], [0,pxh,0], 1, 'FaceColor', (red+2*white)/3, 'EdgeColor', 'none'); % We plot the input PDF.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(xb, mub, 2*sqrt(Sbb), 'o', 'Color', black); % We plot the linearization point.
errorbar(xh, fh, sfm, 'o', 'Color', red); % We plot the measurement point.
axis([minX,maxX,minY,maxY]); % We make sure the axes are set up for the right data range.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey areas.
if exportFigs ~= 0
	export_fig('StochasticMeasurementLinearizationPrior.png','-transparent');
end

% Now we calculate the posterior distribution of the measured input point.
Sxp = inv(dmub'/Shbb*dmub + Sx);
xhp = xh + Sxp*(dmub'/Shbb*(dmub*(xb - xh) + (fh - mub)));
px = 1/sqrt(det(2*pi*Sxp))*exp(-1/2*(Xs - xhp).^2/Sxp);

% We plot the results. That is, we both plot the GP with measurement noise incorporated in its predictions, as well as the posterior probability density of the measurement input point.
figure(6);
clf(6);
hold on;
grid on;
xlabel('Input');
ylabel(['\color[rgb]{',num2str(red(1)),',',num2str(red(2)),',',num2str(red(3)),'}Input probability density\color{black} / \color[rgb]{',num2str(blue(1)),',',num2str(blue(2)),',',num2str(blue(3)),'}Gaussian process output']);
patch([Xs, fliplr(Xs)],[mPost-2*smPost; flipud(mPost+2*smPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-smPost; flipud(mPost+smPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([minX,Xs,maxX], [0,px,0], 1, 'FaceColor', (red+2*white)/3, 'EdgeColor', 'none'); % We plot the input PDF.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(xb, mub, 2*sqrt(Sbb), 'o', 'Color', black); % We plot the linearization point.
plot(xh, fh, 'o', 'Color', red); % We plot the measurement point.
plot([minX,maxX], [fh,fh], '-', 'Color', red); % We plot the horizontal line through the measurement point.
axis([minX,maxX,minY,maxY]); % We make sure the axes are set up for the right data range.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey areas.
if exportFigs ~= 0
	export_fig('StochasticMeasurementLinearizationPosterior.png','-transparent');
end