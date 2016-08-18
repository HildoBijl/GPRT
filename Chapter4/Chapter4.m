% This file contains all the scripts for Chapter 4 of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then first run this block, which initializes many parameters. Subsequently, you can run any block within
% this file separately, or you can just run them all together, for instance by pressing F5 or calling Chapter4 from the Matlab command.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
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
else
	red = [0.8 0 0];
	green = [0 0.4 0];
	blue = [0 0 0.8];
	yellow = [0.6 0.6 0];
	grey = [0.8 0.8 1];
end

% We define data.
lf = 1; % This is the output length scale.
lx = 1; % This is the input length scale.
sfm = 0.1; % This is the output noise scale.
minX = -2; % This is the lower bound of the input space.
maxX = 2; % This is the upper bound of the input space.
Xs = minX:0.01:maxX; % These are the trial points.
ns = length(Xs); % This is the number of trial points.
nm = 10; % How many measurements do we use?
nu = 4; % How many inducing input points do we use?

% We set up measurements.
rng(6, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
Xm = sort(minX + rand(1,nm)*(maxX - minX)); % These are random measurements picked within the interval. We sort them to get our individual FITC plots nicely ordered too.
fmh = sin(2*pi*Xm/4)' + sfm*randn(nm,1); % These are measurement of the function, corrupted by noise.

% We set up the inducing input points.
inputSpaceLength = maxX - minX;
Xu = linspace(minX + (maxX - minX)/(2*nu), maxX - (maxX - minX)/(2*nu), nu); % We divide the points evenly over the input space.

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
Sfm = sfm^2*eye(nm); % This is the noise covariance matrix.

%% Figure 4.1.
disp('Creating Figure 4.1.');

% We first apply Gaussian process regression in the normal way.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results.
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
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('RegularGP.png','-transparent');
end

% We now apply regression to calculate the distribution of the inducing input points.
muu = mu + Kum/(Kmm + Sfm)*(fmh - mm);
Suu = Kuu - Kum/(Kmm + Sfm)*Kmu;
su = sqrt(diag(Suu));
SparseGPSuu = Suu; % We save the covariance matrix. We will need it later on.

% And we plot the results.
figure(2);
clf(2);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('SparseGPTraining.png','-transparent');
end

% Next, we use the inducing input point distribution to calculate the posterior distribution.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results.
figure(3);
clf(3);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('SparseGPPrediction.png','-transparent');
end

%% Figure 4.2.
disp('Creating Figure 4.2.');

% We calculate the posterior distribution based on one measurement point. We do this several times. At the same time, we also calculate the merging of all the distributions which we get like
% this, and we apply online FITC at the same time.
numIndividualPlots = nm;
muuMerged = Kuu\mu; % This is for calculating the posterior distribution of f_u through merging.
SuuInvMerged = inv(Kuu);
muuOnline = mu; % This is for calculating the posterior distribution of f_u through the online update equations.
SuuOnline = Kuu;
SuuStorage = zeros(nu,nu,nm); % We set up a storage parameter for Suu.
for i = 1:numIndividualPlots
	% We apply regression to calculate the distribution of the inducing input points.
	muu = mu + Kum(:,i)/(Kmm(i,i) + Sfm(i,i))*(fmh(i) - mm(i));
	Suu = Kuu - Kum(:,i)/(Kmm(i,i) + Sfm(i,i))*Kmu(i,:);
	su = sqrt(diag(Suu));
	
	% We merge the distribution of muu which we obtained into our previous results.
	SuuInvMerged = SuuInvMerged + inv(Suu) - inv(Kuu);
	muuMerged = muuMerged + Suu\muu - Kuu\mu;
	
	% We apply the online FITC equations. (We don't use them, because we have about four different ways of calculating the posterior distribution of f_u. But this is just to show that they work.)
	Spp = Kmm(i,i) - Kmu(i,:)/Kuu*(Kuu - SuuOnline)/Kuu*Kum(:,i);
	Sup = SuuOnline/Kuu*Kum(:,i);
	Spu = Sup';
	mup = mm(i) + Kmu(i,:)/Kuu*(muuOnline - mu);
	muuOnline = muuOnline + Sup/(Spp + Sfm(i,i))*(fmh(i) - mup);
	SuuOnline = SuuOnline - Sup/(Spp + Sfm(i,i))*Spu;
	SuuStorage(:,:,i) = SuuOnline; % We store the current value of Suu.
	
	% And we plot the results.
	figure(3 + i);
	clf(3 + i);
	hold on;
	grid on;
	xlabel('Input');
	ylabel('Output');
	plot(Xm(i), fmh(i), 'o', 'Color', red); % We plot the measurement points.
	errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
	axis([minX,maxX,-2,2]);
	if exportFigs ~= 0
		export_fig(['FITCIndividualPrediction',num2str(i),'.png'],'-transparent');
	end
end

% We finalize the merged distribution of f_u.
Suu = inv(SuuInvMerged);
muu = Suu*muuMerged;

% We calculate the distribution of f_u through the FITC equations. These should give the same result as the parameters above, but sometimes there may be small differences due to numerical issues,
% especially when the data sets are larger. In that case, the equations below are generally more accurate.
Lmm = diag(diag(Kmm - Kmu/Kuu*Kum));
Duu = Kuu + Kum/(Lmm + Sfm)*Kmu;
muu = mu + Kuu/Duu*Kum/(Lmm + Sfm)*(fmh - mm);
Suu = Kuu/Duu*Kuu;

% Once more, we calculate the distribution of f_u in a different way, but we once more apply tricks to reduce numerical inaccuracies.
KumOLmmSfm = Kum*diag(1./(diag(Kmm - Kmu/Kuu*Kum) + diag(Sfm))); % This is K_um/(Lambda_mm + Sfm). We use our own manual matrix inverse of a diagonal matrix, since sometimes Matlab is not fully accurate when doing this on its own.
KuuODuu = Kuu/(Kuu + KumOLmmSfm*Kmu); % This is K_uu/Delta_uu. We store it because we need it multiple times.
muu = mu + KuuODuu*KumOLmmSfm*(fmh - mm);
Suu = KuuODuu*Kuu;
FITCSuu = Suu; % We store the covariance matrix, in case we want to do extra comparisons.
if exist('SparseGPSuu','var')
	meanStdIncrease = sqrt((det(FITCSuu)/det(SparseGPSuu))^(1/nu));
	disp(['The average increase in the standard deviation, due to the FITC algorithm, is ',num2str(meanStdIncrease),' (factor) = ',num2str((meanStdIncrease-1)*100),'%.']);
end

% And we calculate the standard deviation of the inducing input function values.
su = sqrt(diag(Suu));

% And we plot the results.
figure(14);
clf(14);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
% plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-2,2]);
if exportFigs ~= 0
	export_fig('FITCMergedDistributions.png','-transparent');
end

% Next, we use the inducing input point distribution to calculate the posterior distribution.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results.
figure(15);
clf(15);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-2,2]);
if exportFigs ~= 0
	export_fig('FITCPrediction.png','-transparent');
end

%% Figure 4.3.
disp('Creating Figure 4.3.');

% We calculate the posterior distribution based on a subgroup of measurement points. We do this several times. At the same time, we also calculate the merging of all the distributions which we
% get like this.
groups = {1:floor(nm/4),floor(nm/4)+1:floor(nm/2),floor(nm/2)+1:ceil(nm*3/4),ceil(nm*3/4)+1:nm};
numIndividualPlots = length(groups);
muuMerged = Kuu\mu; % This is for calculating the posterior distribution of f_u through merging.
SuuInvMerged = inv(Kuu);
muuOnline = mu; % This is for calculating the posterior distribution of f_u through the online update equations.
SuuOnline = Kuu;
for i = 1:numIndividualPlots
	% We apply regression to calculate the distribution of the inducing input points.
	ind = groups{i}; % We define which points (indices) we will use in this iteration.
	muu = mu + Kum(:,ind)/(Kmm(ind,ind) + Sfm(ind,ind))*(fmh(ind) - mm(ind));
	Suu = Kuu - Kum(:,ind)/(Kmm(ind,ind) + Sfm(ind,ind))*Kmu(ind,:);
	su = sqrt(diag(Suu));
	
	% We merge the distribution of muu which we obtained into our previous results.
	SuuInvMerged = SuuInvMerged + inv(Suu) - inv(Kuu);
	muuMerged = muuMerged + Suu\muu - Kuu\mu;
	
	% We apply the online PITC equations. (We don't use them, because we have about four different ways of calculating the posterior distribution of fu. But this is just to show that they work.)
	Spp = Kmm(ind,ind) - Kmu(ind,:)/Kuu*(Kuu - SuuOnline)/Kuu*Kum(:,ind);
	Sup = SuuOnline/Kuu*Kum(:,ind);
	Spu = Sup';
	mup = mm(ind) + Kmu(ind,:)/Kuu*(muuOnline - mu);
	muuOnline = muuOnline + Sup/(Spp + Sfm(ind,ind))*(fmh(ind) - mup);
	SuuOnline = SuuOnline - Sup/(Spp + Sfm(ind,ind))*Spu;
	
	% And we plot the results.
	figure(15 + i);
	clf(15 + i);
	hold on;
	grid on;
	xlabel('Input');
	ylabel('Output');
	plot(Xm(ind), fmh(ind), 'o', 'Color', red); % We plot the measurement points.
	errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
	axis([minX,maxX,-2,2]);
	if exportFigs ~= 0
		export_fig(['PITCIndividualPrediction',num2str(i),'.png'],'-transparent');
	end
end

% We finalize the merged distribution of f_u.
Suu = inv(SuuInvMerged);
muu = Suu*muuMerged;
PITCSuu = Suu; % We store the covariance matrix, in case we want to do extra comparisons.
if exist('SparseGPSuu','var')
	meanStdIncrease = sqrt((det(PITCSuu)/det(SparseGPSuu))^(1/nu));
	disp(['The average increase in the standard deviation, due to the PITC algorithm, is ',num2str(meanStdIncrease),' (factor) = ',num2str((meanStdIncrease-1)*100),'%.']);
end
	
% We calculate the distribution of f_u through the PITC equations. These should give the same result as the parameters above, but sometimes there may be small differences due to numerical issues,
% especially when the data sets are larger. In that case, the equations below are generally more accurate.
LmmStart = Kmm - Kmu/Kuu*Kum; % This is the Lambda_mm matrix before the block-diagonalization.
Lmm = blkdiag(LmmStart(groups{1},groups{1}),LmmStart(groups{2},groups{2}),LmmStart(groups{3},groups{3}),LmmStart(groups{4},groups{4})); % This is the Lambda_mm matrix for PITC.
Duu = Kuu + Kum/(Lmm + Sfm)*Kmu;
muu = mu + Kuu/Duu*Kum/(Lmm + Sfm)*(fmh - mm);
Suu = Kuu/Duu*Kuu;

% Once more, we calculate the distribution of f_u in a different way, but we once more apply tricks to reduce numerical inaccuracies.
LmmSfm = LmmStart + Sfm;
KumOLmmSfm = [Kum(:,groups{1})/LmmSfm(groups{1},groups{1}),Kum(:,groups{2})/LmmSfm(groups{2},groups{2}),Kum(:,groups{3})/LmmSfm(groups{3},groups{3}),Kum(:,groups{4})/LmmSfm(groups{4},groups{4})];
KuuODuu = Kuu/(Kuu + KumOLmmSfm*Kmu); % This is K_uu/Delta_uu. We store it because we need it multiple times.
muu = mu + KuuODuu*KumOLmmSfm*(fmh - mm);
Suu = KuuODuu*Kuu;

% We also calculate the posterior distribution of f_u by adding a single measurement at a time, but still adding them to the respective group. (This is the second updating method we looked at in
% the thesis.) In this case it is obviously not a sensible way of setting up the posterior distribution, but it does show how the equations can be implemented. We won't use these results further
% though, as we already have plenty of methods of calculating the posterior distribution of f_u.
muuOnline = mu;
SuuOnline = Kuu;
Lmm = Kmm - Kmu/Kuu*Kum; % This is \Lambda_{mm}. It is the same for all points, so we calculate it only once.
for i = 1:length(groups) % We first walk through the groups.
	group = groups{i};
	for j = 1:length(group) % We then walk through the points within the group.
		% We define some indices which we will need.
		ind = group(j); % This is the index of the point we will add.
		prevInd = group(1:j-1); % These are the indices of the already processed points within the group. We need to take them into account.
		
		% We define helpful matrices.
		Kpp = Kmm(ind,ind);
		Kpm = Kmm(ind,:);
		Kmp = Kpm';
		Kpu = Kmu(ind,:);
		Kup = Kpu';
		Lpp = Kpp - Kpu/Kuu*Kup;
		Lpm = Kpm - Kpu/Kuu*Kum;
		Lmp = Lpm';
		Ltpp = Lpp - Lpm(:,prevInd)/(Lmm(prevInd,prevInd) + Sfm(prevInd,prevInd))*Lmp(prevInd,:); % This is \tilde{\Lambda}_{++}.
		Ktup = Kup - Kum(:,prevInd)/(Lmm(prevInd,prevInd) + Sfm(prevInd,prevInd))*Lmp(prevInd,:); % This is \tilde{K}_{+u}.
		Ktpu = Ktup';
		mp = mm(ind);
		mtp = mp + Lpm(:,prevInd)/(Lmm(prevInd,prevInd) + Sfm(prevInd,prevInd))*(fmh(prevInd) - mm(prevInd)); % This is \tilde{m}_+.
		
		% We apply the online PITC update equations.
		Spp = Ltpp + Ktpu/Kuu*SuuOnline/Kuu*Ktup;
		Sup = SuuOnline/Kuu*Ktup;
		Spu = Sup';
		mup = mtp + Ktpu/Kuu*(muuOnline - mu);
		muuOnline = muuOnline + Sup/(Spp + Sfm(i,i))*(fmh(i) - mup); % Note that this equation is exactly the same as for the FITC update. How elegant.
		SuuOnline = SuuOnline - Sup/(Spp + Sfm(i,i))*Spu; % Note that this equation is exactly the same as for the FITC update. How elegant.
	end
end

% And we calculate the standard deviation of the inducing input function values.
su = sqrt(diag(Suu));

% And we plot the results.
figure(20);
clf(20);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
% plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-2,2]);
if exportFigs ~= 0
	export_fig('PITCMergedDistributions.png','-transparent');
end

% Next, we use the inducing input point distribution to calculate the posterior distribution.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results.
figure(21);
clf(21);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-2,2]);
if exportFigs ~= 0
	export_fig('PITCPrediction.png','-transparent');
end

%% Figure 4.4.
disp('Creating Figure 4.4.');

% We set up the posterior distribution of the weights.
% w = Kuu\(muu - mu); % These are the true values of the weights, although we will not use them. To be able to run this line, you need to run another block to calculate muu first.
Kw = 1e3*eye(nu); % This is the prior covariance of the weights. I manually chose it. Choosing it too small will distort results, so I chose it big enough not to have any significant effect. Feel free to vary it and check out what happens though.
Sw = inv(Kum/Sfm*Kmu + inv(Kw)); % This is the posterior weight covariance matrix.
muw = Sw*Kum/Sfm*fmh; % This is the posterior weight mean matrix.
mPost = Ksu*muw; % This is the posterior mean of the trial points.
SPost = Ksu*Sw*Kus; % This is the posterior covariance matrix of the trial points.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.

% We plot the results.
figure(22);
clf(22);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('IncorrectSparseGPRegression.png','-transparent');
end

%% Figure 4.5.
% For this block to work, you should have already run the block for Figure 4.2, which generated learning data for the online FITC algorithm.
disp('Creating Figure 4.5.');

% We calculate the learning indices based on stored values of the covariance matrix \Sigma_{uu} and plot them.
learningIndices = ones(nu,nm+1); % We set up the initial values of the learning indices.
for i = 1:nm
	learningIndices(:,i+1) = sqrt(diag(SuuStorage(:,:,i))./diag(Kuu));
end
colors = {red,blue,yellow,green};
figure(23);
clf(23);
hold on;
grid on;
xlabel('Number of measurements implemented');
ylabel('Learning index');
for i = 1:nu
	plot(0:nm,learningIndices(i,:),'-','Color',colors{i});
end
legend(['Point ',num2str(Xu(1))],['Point ',num2str(Xu(2))],['Point ',num2str(Xu(3))],['Point ',num2str(Xu(4))]);
if exportFigs ~= 0
	export_fig('LearningIndices.png','-transparent');
end

%% Figure 4.6.
disp('Creating Figure 4.6.');

% We will optimize the inducing input points. We first find the ones for the FITC algorithm that best explain the data that we measured.
Xu = sort((minX + maxX)/2 + (1/4)*(maxX - minX)*randn(1,nu)); % We pick random initial inducing input points.
options = optimset('Display', 'off') ; % We do not want any output from fmincon.
Xu = fmincon(@(Xu)( ...
	logdet(diag(diag(Kmm + Sfm - (lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2))/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*(lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2)))) + (lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2))/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*(lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2))) ...
	+ 1/2*(fmh - mm)'/(diag(diag(Kmm + Sfm - (lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2))/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*(lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2)))) + (lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2))/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*(lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2)))*(fmh - mm) ...
), Xu, [], [], [], [], [], [], [], options); % We optimize the probability that we obtained the measurements that we obtained. Note that we want to maximize the log-likelihood, so we minimize minus the log-likelihood.
disp(['When optimizing the likelihood, we found the following inducing input points.']);
disp(sort(Xu));

% We adjust the covariance matrices, based on the new inducing input set Xu.
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
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.

% Next, we use the new inducing input point distribution to calculate the posterior distribution. (We use the sparse GP algorithm because it has easier equations and to make a fair comparison.)
muu = mu + Kum/(Kmm + Sfm)*(fmh - mm);
Suu = Kuu - Kum/(Kmm + Sfm)*Kmu;
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
su = sqrt(diag(Suu)); % These are the posterior standard deviations of the inducing function values.

% We plot the results.
figure(24);
clf(24);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('OptimizedInducingInputPointsThroughLikelihood.png','-transparent');
end

% Next, we tune the inducing inputs again, but now based on a given trial function set X_*. For this set, we optimize Kss - Ksu/Kuu*Kum/(Kmm + Sfm)*Kmu/Kuu*Kus
nsOptimization = 9; % How many trial points shall we use?
XsOptimization = linspace(minX, maxX, nsOptimization); % This is the set of trial points which we will use.
nu = 4; % How many inducing input points shall we tune?
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
Xu = sort((minX + maxX)/2 + (1/4)*(maxX - minX)*randn(1,nu)); % We pick random initial inducing input points.
Kss = lf^2*exp(-1/2*(repmat(XsOptimization,nsOptimization,1) - repmat(XsOptimization',1,nsOptimization)).^2/lx^2);
% Xu = fmincon(@(Xu)(sum(log(diag(Kss - lf^2*exp(-1/2*(repmat(Xu,nsOptimization,1) - repmat(XsOptimization',1,nu)).^2/lx^2)/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2)/(Kmm + Sfm)*lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2)/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*lf^2*exp(-1/2*(repmat(XsOptimization,nu,1) - repmat(Xu',1,nsOptimization)).^2/lx^2))))), Xu, [], [], [], [], [], [], [], options); % This is an old expression. It minimizes the product of the diagonal elements of \Sigma_{**}. It works, but minimizing the entropy seems more sensible. By the way, we don't just minimize the product of the diagonal elements of \Sigma_{**} but actually the logarithm of this quantity. And to make things computationally more stable, we do not take the logarithm of the product, but the sum of the logarithms, which is the same.
Xu = fmincon(@(Xu)(logdet(Kss - lf^2*exp(-1/2*(repmat(Xu,nsOptimization,1) - repmat(XsOptimization',1,nu)).^2/lx^2)/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2)/(Kmm + Sfm)*lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2)/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))*lf^2*exp(-1/2*(repmat(XsOptimization,nu,1) - repmat(Xu',1,nsOptimization)).^2/lx^2))), Xu, [], [], [], [], [], [], [], options); % Here we minimize the logarithm of the determinant (which is proportional to the entropy) of \Sigma_{**}. 
disp(['For a given X_* with ',num2str(nsOptimization),' trial points, we found the following inducing input points.']);
disp(sort(Xu));

% We adjust the covariance matrices, based on the new inducing input set Xu.
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
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.

% Next, we use the new inducing input point distribution to calculate the posterior distribution.
muu = mu + Kum/(Kmm + Sfm)*(fmh - mm);
Suu = Kuu - Kum/(Kmm + Sfm)*Kmu;
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
su = sqrt(diag(Suu)); % These are the posterior standard deviations of the inducing function values.

% We plot the results.
figure(25);
clf(25);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('OptimizedInducingInputPointsThroughSet.png','-transparent');
end

% We once more optimize the inducing input points, but now do so using a distribution for the trial points.
muxs = 0; % This is the mean of the trial input points.
Sxs = ((maxX - minX)/4)^2; % This is the covariance of the trial input points. The bigger this range, the more the inducing input points will be spread out.
Xu = sort((minX + maxX)/2 + (1/4)*(maxX - minX)*randn(1,nu)); % We pick random initial inducing input points.
Xu = fmincon(@(Xu)(lf^2 - sqrt(lx^2/(2*Sxs + lx^2))*sum(sum( ...
	((lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))\(lf^2*exp(-1/2*(repmat(Xm,nu,1) - repmat(Xu',1,nm)).^2/lx^2))/(Kmm + Sfm)*(lf^2*exp(-1/2*(repmat(Xu,nm,1) - repmat(Xm',1,nu)).^2/lx^2))/(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/lx^2))).* ...
	(lf^2*exp(-1/2*(repmat(Xu,nu,1) - repmat(Xu',1,nu)).^2/(2*lx^2))).* ...
	(lf^2*exp(-1/2*((repmat(Xu,nu,1) + repmat(Xu',1,nu))/2 - muxs).^2/(lx^2/2 + Sxs))) ...
))), Xu, [], [], [], [], [], [], [], options); % We optimize the solution of the integral over all possible values of x_*.
disp(['For x_* ~ \N(0,',num2str(Sxs),'), we found the following inducing input points.']);
disp(sort(Xu));

% We adjust the covariance matrices, based on the new inducing input set Xu.
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
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.

% Next, we use the new inducing input point distribution to calculate the posterior distribution.
muu = mu + Kum/(Kmm + Sfm)*(fmh - mm);
Suu = Kuu - Kum/(Kmm + Sfm)*Kmu;
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
su = sqrt(diag(Suu)); % These are the posterior standard deviations of the inducing function values.

% We plot the results.
figure(26);
clf(26);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
plot(Xm, fmh, 'o', 'Color', red); % We plot the measurement points.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('OptimizedInducingInputPointsThroughIntegral.png','-transparent');
end

%% Figure 4.7.
disp('Creating Figure 4.7.');

% We reset the inducing input points to their usual values, in case they changed, and recalculate covariance matrices.
nu = 4;
Xu = linspace(minX + (maxX - minX)/(2*nu), maxX - (maxX - minX)/(2*nu), nu);
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
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.

% We make a sparse GP prediction of the inducing input point distribution.
muu = mu + Kum/(Kmm + Sfm)*(fmh - mm);
Suu = Kuu - Kum/(Kmm + Sfm)*Kmu;

% Before we change the inducing input points, we make a `before' prediction which we will also plot.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
su = sqrt(diag(Suu)); % These are the posterior standard deviations of the inducing function values
figure(27);
clf(27);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('ShiftedInducingInputPointDistributionBefore.png','-transparent');
end

% We set up some extra inducing input points and find the corresponding covariance matrices.
Xv = linspace(minX + (maxX - minX)/nu, maxX - (maxX - minX)/nu, nu-1);
nv = size(Xv,2);
X = [Xu,Xv];
n = size(X,2);
diff = repmat(X,n,1) - repmat(X',1,n); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*diff.^2/lx^2); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,nv],[nu,nv]);
Kuu = KDivided{1,1};
Kuv = KDivided{1,2};
Kvu = KDivided{2,1};
Kvv = KDivided{2,2};
mv = zeros(nv,1); % This is the prior mean vector of the new inducing input points.

% We set up the joint distribution for the full set of inducing input points.
jointS = [Suu,Suu/Kuu*Kuv;Kvu/Kuu*Suu,Kvv - Kvu/Kuu*(Kuu - Suu)/Kuu*Kuv];
jointMu = [muu;mv + Kvu/Kuu*(muu - mu)];

% We now indicate which inducing input points we want and adjust the corresponding parameters. This is where we can cut certain inducing input points.
indices = [1:nu-1,nu+1:nu+nv];
Xu = X(:,indices);
Suu = jointS(indices,indices);
muu = jointMu(indices);
nu = size(Xu,2);

% We adjust the covariance matrices, based on the new inducing input set Xu.
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
mu = zeros(nu,1); % This is the prior mean vector of the inducing input points.

% Next, we use the new inducing input point distribution to calculate the posterior distribution.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
su = sqrt(diag(Suu)); % These are the posterior standard deviations of the inducing function values.

% We plot the results.
figure(28);
clf(28);
hold on;
grid on;
xlabel('Input');
ylabel('Output');
patch([Xs, fliplr(Xs)],[mPost-2*sPost; flipud(mPost+2*sPost)], 1, 'FaceColor', (grey+white)/2, 'EdgeColor', 'none'); % This is the grey area in the plot.
patch([Xs, fliplr(Xs)],[mPost-sPost; flipud(mPost+sPost)], 1, 'FaceColor', grey, 'EdgeColor', 'none'); % This is the grey area in the plot.
set(gca, 'layer', 'top'); % We make sure that the grid lines and axes are above the grey area.
plot(Xs, mPost, '-', 'LineWidth', 1, 'Color', blue); % We plot the mean line.
errorbar(Xu,muu,2*su,'*','Color',yellow); % We plot the inducing input points.
axis([minX,maxX,-1.5,1.5]);
if exportFigs ~= 0
	export_fig('ShiftedInducingInputPointDistributionAfter.png','-transparent');
end