% This file contains the experiments for Chapter 4 of the Gaussian process regression thesis related to the performance of the various algorithms. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then you can run this file using ctrl+enter or F5.
% In this file, we will first apply regular GP, online PITC, offline PITC, online FITC and offline FITC, in that order.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.

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

% We define settings for the script.
nmTotal = 1e6; % This is the number of measurements we will set up in total. We may not need all of them.
nmPlot = 50; % This is the number of measurements used to make the plots.
nm = 1e4; % This is the number of measurements used by the GP regression algorithm.
nsPerDimension = 21; % This is the number of trial points per dimension. These are also used to calculate the RMSE.
nuPerDimension = 5; % This is the number of inducing input points for each dimension. So the total number used will be the square of this number.
lf = 1; % This is the output length scale of the Gaussian process.
lx = [1;1]; % This is the input length scale of the Gaussian process for each input dimension.
sfm = 0.5; % This is the noise standard deviation applied to each of the dimensions.
groupSize = nuPerDimension^2; % This is the group size used for the PITC algorithm.

% We calculate the trial function for a variety of points.
xMin = [-2;-2]; % These are the minimum values for x.
xMax = [2;2]; % These are the maximum values for x.
ns = nsPerDimension^2; % This is the total number of trial points.
[x1Mesh,x2Mesh] = meshgrid(linspace(xMin(1),xMax(1),nsPerDimension),linspace(xMin(2),xMax(2),nsPerDimension));
Xs = [reshape(x1Mesh,1,ns); reshape(x2Mesh,1,ns)];
fs = reshape(trialFunction(Xs),nsPerDimension,nsPerDimension);

% We set up the inducing input points for all algorithms.
nu = nuPerDimension^2;
[xu1Mesh,xu2Mesh] = meshgrid(linspace(xMin(1),xMax(1),nuPerDimension),linspace(xMin(2),xMax(2),nuPerDimension));
Xu = [reshape(xu1Mesh,1,nu); reshape(xu2Mesh,1,nu)];

% And then we plot the result.
figure(1);
clf(1);
hold on;
grid on;
sFunc = surface(x1Mesh, x2Mesh, fs);
xlabel('x_1');
ylabel('x_2');
zlabel('f(x_1,x_2)');
view([60,35]);
if exportFigs ~= 0
	export_fig('TrialFunction.png','-transparent');
end

% Next, we generate measurements from the trial function. We make sure to have enough measurements to last for a while.
disp('Setting up measurement data.');
Xm = repmat(xMin,1,nmTotal) + repmat(xMax - xMin,1,nmTotal).*rand(2,nmTotal);
fm = trialFunction(Xm)';
fmh = fm + randn(size(fm))*sfm;

%% We now use the first couple of measurements to generate a GP regression plot. We set up a GP for this.
disp(['Setting up GP regression for ',num2str(nmPlot),' measurements.']);
X = [Xm(:,1:nmPlot),Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nmPlot,ns],[nmPlot,ns]);
Kmm = KDivided{1,1};
Kms = KDivided{1,2};
Ksm = KDivided{2,1};
Kss = KDivided{2,2};
Sfm = sfm^2*eye(nmPlot);
mm = zeros(nmPlot,1);
ms = zeros(ns,1);

% We apply Gaussian process regression in the normal way.
mPost = ms + Ksm/(Kmm + Sfm)*(fmh(1:nmPlot) - mm); % This is the posterior mean vector.
SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

% We plot the results.
figure(2);
clf(2);
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
scatter3(Xm(1,1:nmPlot), Xm(2,1:nmPlot), fmh(1:nmPlot)', 'ro', 'filled');
sFunc = surface(x1Mesh, x2Mesh, fs);
set(sFunc,'FaceAlpha',0.8);
set(sFunc,'FaceColor',green);
xlabel('x_1');
ylabel('x_2');
zlabel('f(x_1,x_2)');
view([60,35]);
axis([xMin(1),xMax(1),xMin(2),xMax(2),-3,2]);
if exportFigs ~= 0
	export_fig(['TrialFunctionApproximationWith',num2str(nmPlot),'Measurements.png'],'-transparent');
end

% And we plot the error.
figure(3);
clf(3);
hold on;
grid on;
sDown = surface(x1Mesh, x2Mesh, mPost - fs - 2*sPost);
set(sDown,'FaceAlpha',0.3);
set(sDown,'LineStyle','none');
set(sDown,'FaceColor',blue);
sUp = surface(x1Mesh, x2Mesh, mPost - fs + 2*sPost);
set(sUp,'FaceAlpha',0.3);
set(sUp,'LineStyle','none');
set(sUp,'FaceColor',blue);
sMid = surface(x1Mesh, x2Mesh, mPost - fs);
set(sMid,'FaceAlpha',0.8);
set(sMid,'FaceColor',blue);
scatter3(Xm(1,1:nmPlot), Xm(2,1:nmPlot), fmh(1:nmPlot)' - fm(1:nmPlot)', 'ro', 'filled');
sFunc = surface(x1Mesh, x2Mesh, fs - fs);
set(sFunc,'FaceAlpha',0.8);
set(sFunc,'FaceColor',green);
xlabel('x_1');
ylabel('x_2');
zlabel('f(x_1,x_2)');
view([60,3]);
axis([xMin(1),xMax(1),xMin(2),xMax(2),-2,2]);
if exportFigs ~= 0
	export_fig(['TrialFunctionErrorWith',num2str(nmPlot),'Measurements.png'],'-transparent');
end

% We calculate the root mean squared error for the experiment we just did.
RMSE = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The prediction RMSE for ',num2str(nmPlot),' measurement points is ',num2str(RMSE),'.']);

%% Next, we start doing the real experiments. We start with GP.
% We set up the covariance and make the predictions.
disp('Starting experiment 1...');
tic;
X = [Xm(:,1:nm),Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nm,ns],[nm,ns]);
Kmm = KDivided{1,1};
Kms = KDivided{1,2};
Ksm = KDivided{2,1};
Kss = KDivided{2,2};
Sfm = sfm^2*eye(nm);
mm = zeros(nm,1);
ms = zeros(ns,1);
KsmDivKmm = Ksm/(Kmm + Sfm); % We calculate this quantity only once.
t1 = toc; % Having finished the training stage counts as being done.
mPost = ms + KsmDivKmm*(fmh(1:nm) - mm); % This is the posterior mean vector.
SPost = Kss - KsmDivKmm*Kms; % This is the posterior covariance matrix.
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE1 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The regular GP regression algorithm used ',num2str(nm),' measurements in ',num2str(t1),' seconds. The RMSE was ',num2str(RMSE1),'.']);

%% This block contains the online PITC algorithm, once for nm measurements and once until t1 time.

% We start with PITC applied to nm experiments (case 2).
disp('Starting experiment 2 (online)...');
tic;

% We set up covariance matrices.
X = [Xu,Xs]; % We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Sfm = sfm^2*ones(nm,1);
mm = zeros(nm,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We run the online PITC algorithm. We walk through the groups, adding them one by one in an online way.
Suu = Kuu;
muu = mu;
groupSizes = groupSize*ones(1,ceil(nm/groupSize)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
groupSizes(end) = mod(nm-1, groupSize)+1;
groups = mat2cell(1:nm, 1, groupSizes);
for i = 1:length(groups);
	indices = groups{i}; % We look up which measurements we use during this iteration.
	Kpp = lf^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx.^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is K++. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kup = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,length(indices),1]),3)); % This is Ku+. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kpu = Kup';
	KpuDivKuu = Kpu/Kuu; % This is K+u/Kuu.
	Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is \Sigma_{++}.
	Spu = KpuDivKuu*Suu; % This is \Sigma_{+u}.
	Sup = Spu';
	SupDivSpp = Sup/(Spp + diag(Sfm(indices)));
	mup = mm(indices) + Kpu/Kuu*(muu - mu); % This is \mu_+.
	muu = muu + SupDivSpp*(fmh(indices) - mup); % This is the update law for \mu_u.
	Suu = Suu - SupDivSpp*Spu; % This is the update law for \Sigma_{uu}.
end
t2Online = toc; % Having finished the training stage counts as being done, so we record the time.

% We calculate the posterior distribution of the trial points.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE2 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The online PITC regression algorithm used ',num2str(nm),' measurements in ',num2str(t2Online),' seconds. The RMSE was ',num2str(RMSE2),'.']);

% We now run the PITC for the same time as the online GP algorithm (case 3).
disp('Starting experiment 3 (online)...');
tic;

% We set up covariance matrices.
X = [Xu,Xs]; % We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
mu = zeros(nu,1);
ms = zeros(ns,1);

% We run the online PITC algorithm. We walk through the groups, adding them one by one in an online way, until time runs out.
Suu = Kuu;
muu = mu;
i = 0;
while toc < t1
	i = i + 1;
	indices = (i-1)*groupSize+1:i*groupSize; % We define which measurements we use during this iteration.
	Kpp = lf^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx.^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is K++. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kup = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,length(indices),1]),3)); % This is Ku+. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kpu = Kup';
	KpuDivKuu = Kpu/Kuu; % This is K+u/Kuu.
	Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is \Sigma_{++}.
	Spu = KpuDivKuu*Suu; % This is \Sigma_{+u}.
	Sup = Spu';
	SupDivSpp = Sup/(Spp + sfm^2*eye(groupSize));
	mup = zeros(groupSize,1) + Kpu/Kuu*(muu - mu); % This is \mu_+.
	muu = muu + SupDivSpp*(fmh(indices) - mup); % This is the update law for \mu_u.
	Suu = Suu - SupDivSpp*Spu; % This is the update law for \Sigma_{uu}.
end
nm3 = i*groupSize; % We record the number of measurements used by the online PITC algorithm.

% We calculate the posterior distribution of the trial points.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE3 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The online PITC regression algorithm used ',num2str(nm3),' measurements in ',num2str(t1),' seconds. The RMSE was ',num2str(RMSE3),'.']);

%% This block contains the offline PITC algorithm, once for nm measurements and once for nm3 measurements.

% We apply the offline PITC algorithm with the same amount of measurements.
disp('Starting experiment 2 (offline)...');
tic;

% We define the groups which we will use.
groupSizes = groupSize*ones(1,ceil(nm/groupSize)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
groupSizes(end) = mod(nm-1, groupSize)+1;
groups = mat2cell(1:nm, 1, groupSizes);

% We set up covariance matrices.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kum = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,nm,1]),3));
Kmu = Kum';
Sfm = sfm^2*ones(nm,1);
mm = zeros(nm,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop.
KumDivLmm = zeros(nu,nm); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
for i = 1:length(groups)
	indices = groups{i}; % These are the indices of the measurements we will use in this group.
	Kmm = lf^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx.^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
	Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
	KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
end

% Finally we apply the offline PITC equations.
Duu = Kuu + KumDivLmm*Kmu;
KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
Suu = KuuDivDuu*Kuu;
muu = KuuDivDuu*KumDivLmm*(fmh(1:nm) - mm);
t2Offline = toc; % Having finished the training stage counts as being done.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE2 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The offline PITC regression algorithm used ',num2str(nm),' measurements in ',num2str(t2Offline),' seconds. The RMSE was ',num2str(RMSE2),'.']);

% We apply the offline PITC algorithm with the same amount of measurements as the online PITC algorithm of case 3.
disp('Starting experiment 3 (offline)...');
tic;

% We define the groups which we will use.
groupSizes = groupSize*ones(1,ceil(nm3/groupSize)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
groupSizes(end) = mod(nm3-1, groupSize)+1;
groups = mat2cell(1:nm3, 1, groupSizes);

% We set up covariance matrices.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kum = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm3]) - repmat(permute(Xm(:,1:nm3),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,nm3,1]),3));
Kmu = Kum';
Sfm = sfm^2*ones(nm3,1);
mm = zeros(nm3,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop. This is why the offline PITC algorithm is slower than you would
% otherwise expect.
KumDivLmm = zeros(nu,nm3); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
for i = 1:length(groups)
	indices = groups{i}; % These are the indices of the measurements we will use in this group.
	Kmm = lf^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx.^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
	Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
	KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
end

% Finally we apply the offline PITC equations.
Duu = Kuu + KumDivLmm*Kmu;
KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
Suu = KuuDivDuu*Kuu;
muu = KuuDivDuu*KumDivLmm*(fmh(1:nm3) - mm);
t2Offline = toc; % Having finished the training stage counts as being done.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE3 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The offline PITC regression algorithm used ',num2str(nm3),' measurements in ',num2str(t2Offline),' seconds. The RMSE was ',num2str(RMSE3),'.']);

%% This block contains the online FITC algorithm, once for nm measurements, once for nm3 measurements and once until we use the same time as experiments 1 and 3.

% We apply the online FITC algorithm with the same amount of measurements as experiments 1 and 2.
disp('Starting experiment 4 (online)...');
tic;

% We set up covariance matrices. Yes, we do this again, to keep the runtime comparison fair.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kuui = (eye(nu)/Kuu); % We already invert Kuu. For some reason inv(Kuu) gives too many numerical problems here, so we have to use this instead. (Oh, the chaos of Matlab... Just try to calculate eye(nu)/Kuu - inv(Kuu). It should be zero, right? Well, it is FAR from zero.)

% We use online updating to calculate the distribution of the inducing function values.
Suu = Kuu;
muu = mu;
for i = 1:nm
	Kpp = lf^2; % This is K++.
	Kup = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,1]) - repmat(permute(Xm(:,i),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,1,1]),3)); % This is Ku+. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kpu = Kup';
	KpuDivKuu = Kpu*Kuui; % This is K+u/Kuu. Note that, if we would precompute inv(Kuu) and use it here, we would save a lot of time, but Matlab would give very wrong results due to numerical problems.
	Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is \Sigma_{++}.
	Spu = KpuDivKuu*Suu; % This is \Sigma_{+u}.
	Sup = Spu';
	SupDivSpp = Sup/(Spp + sfm^2);
	mup = 0 + KpuDivKuu*(muu - mu); % This is \mu_+.
	muu = muu + SupDivSpp*(fmh(i) - mup); % This is the update law for \mu_u.
	Suu = Suu - SupDivSpp*Spu; % This is the update law for \Sigma_{uu}.
end
t4Online = toc; % Having finished the training stage counts as being done.

% We calculate the posterior distribution of the trial points.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE4 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The online FITC regression algorithm used ',num2str(nm),' measurements in ',num2str(t4Online),' seconds. The RMSE was ',num2str(RMSE4),'.']);

% We apply the online FITC algorithm with the same amount of measurements as experiment 3.
disp('Starting experiment 5 (online)...');
tic;

% We set up covariance matrices. Yes, we do this again, to keep the runtime comparison fair.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kuui = (eye(nu)/Kuu); % We already invert Kuu. For some reason inv(Kuu) gives too many numerical problems here, so we have to use this instead. (Oh, the chaos of Matlab... Just try to calculate eye(nu)/Kuu - inv(Kuu). It should be zero, right? Well, it is FAR from zero.)

% We use online updating to calculate the distribution of the inducing function values.
Suu = Kuu;
muu = mu;
for i = 1:nm3
	Kpp = lf^2; % This is K++.
	Kup = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,1]) - repmat(permute(Xm(:,i),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,1,1]),3)); % This is Ku+. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kpu = Kup';
	KpuDivKuu = Kpu*Kuui; % This is K+u/Kuu.
	Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is \Sigma_{++}.
	Spu = KpuDivKuu*Suu; % This is \Sigma_{+u}.
	Sup = Spu';
	SupDivSpp = Sup/(Spp + sfm^2);
	mup = 0 + KpuDivKuu*(muu - mu); % This is \mu_+.
	muu = muu + SupDivSpp*(fmh(i) - mup); % This is the update law for \mu_u.
	Suu = Suu - SupDivSpp*Spu; % This is the update law for \Sigma_{uu}.
end
t5Online = toc; % Having finished the training stage counts as being done.

% We calculate the posterior distribution of the trial points.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE5 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The online FITC regression algorithm used ',num2str(nm3),' measurements in ',num2str(t5Online),' seconds. The RMSE was ',num2str(RMSE5),'.']);

% We apply the online FITC algorithm with the same amount of runtime as experiments 1 and 3.
disp('Starting experiment 6 (online)...');
tic;

% We set up covariance matrices. Yes, we do this again, to keep the runtime comparison fair.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kuui = (eye(nu)/Kuu); % We already invert Kuu. For some reason inv(Kuu) gives too many numerical problems here, so we have to use this instead. (Oh, the chaos of Matlab... Just try to calculate eye(nu)/Kuu - inv(Kuu). It should be zero, right? Well, it is FAR from zero.)

% We use online updating to calculate the distribution of the inducing function values.
Suu = Kuu;
muu = mu;
i = 0;
while toc < t1
	i = i + 1;
	Kpp = lf^2; % This is K++.
	Kup = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,1]) - repmat(permute(Xm(:,i),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,1,1]),3)); % This is Ku+. Although it is known from the previous algorithm, we recalculate it to make a fair runtime comparison.
	Kpu = Kup';
	KpuDivKuu = Kpu*Kuui; % This is K+u/Kuu.
	Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is \Sigma_{++}.
	Spu = KpuDivKuu*Suu; % This is \Sigma_{+u}.
	Sup = Spu';
	SupDivSpp = Sup/(Spp + sfm^2);
	mup = 0 + KpuDivKuu*(muu - mu); % This is \mu_+.
	muu = muu + SupDivSpp*(fmh(i) - mup); % This is the update law for \mu_u.
	Suu = Suu - SupDivSpp*Spu; % This is the update law for \Sigma_{uu}.
end
nm6 = i;

% We calculate the posterior distribution of the trial points.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE6 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The online FITC regression algorithm used ',num2str(nm6),' measurements in ',num2str(t1),' seconds. The RMSE was ',num2str(RMSE6),'.']);

%% This block contains the offline FITC algorithm, once for nm measurements, once for nm3 measurements and once for nm6 measurements.

% We apply the offline FITC algorithm with the same amount of measurements as experiments 1 and 2.
disp('Starting experiment 4 (offline)...');
tic;

% We calculate covariance matrices.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kum = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,nm,1]),3));
Kmu = Kum';
Kmm = lf^2*ones(nm,1); % We only take the diagonal elements of the original Kmm matrix here, because these are the only ones which we will need. We do not need the rest.
Qmm = sum((Kmu/Kuu).*Kmu,2); % This is the diagonal of Kmu/Kuu*Kum, but then calculated in a way that takes O(nm*nu^2) time instead of O(nm^2*nu) time.
Lmm = Kmm - Qmm; % This is \Lambda_{mm}. Or at least, its diagonal elements stored in a vector. There's no use storing the full matrix when the matrix is diagonal anyway.
Sfm = sfm^2*ones(nm,1);
mm = zeros(nm,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We apply the offline FITC equations.
KumDivLmm = Kum./repmat((Lmm + Sfm)',nu,1); % We store this parameter because we need it multiple times. We use this method of calculating because it prevents us from having an nm by nm matrix.
Duu = Kuu + KumDivLmm*Kmu;
KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
Suu = KuuDivDuu*Kuu;
muu = KuuDivDuu*KumDivLmm*(fmh(1:nm) - mm);
t4Offline = toc; % Having finished the training stage counts as being done.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE4 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The FITC regression algorithm used ',num2str(nm),' measurements in ',num2str(t4Offline),' seconds. The RMSE was ',num2str(RMSE4),'.']);

% We apply the offline FITC algorithm with the same amount of measurements as experiment 3.
disp('Starting experiment 5 (offline)...');
tic;

% We calculate covariance matrices.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kum = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm3]) - repmat(permute(Xm(:,1:nm3),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,nm3,1]),3));
Kmu = Kum';
Kmm = lf^2*ones(nm3,1); % We only take the diagonal elements of the original Kmm matrix here, because these are the only ones which we will need. We do not need the rest.
Qmm = sum((Kmu/Kuu).*Kmu,2); % This is the diagonal of Kmu/Kuu*Kum, but then calculated in a way that takes O(nm*nu^2) time instead of O(nm^2*nu) time.
Lmm = Kmm - Qmm; % This is \Lambda_{mm}. Or at least, its diagonal elements stored in a vector. There's no use storing the full matrix when the matrix is diagonal anyway.
Sfm = sfm^2*ones(nm3,1);
mm = zeros(nm3,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We apply the offline FITC equations.
KumDivLmm = Kum./repmat((Lmm + Sfm)',nu,1); % We store this parameter because we need it multiple times. We use this method of calculating because it prevents us from having an nm by nm matrix.
Duu = Kuu + KumDivLmm*Kmu;
KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
Suu = KuuDivDuu*Kuu;
muu = KuuDivDuu*KumDivLmm*(fmh(1:nm3) - mm);
t5Offline = toc; % Having finished the training stage counts as being done.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE5 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The FITC regression algorithm used ',num2str(nm3),' measurements in ',num2str(t5Offline),' seconds. The RMSE was ',num2str(RMSE5),'.']);

% We apply the offline FITC algorithm with the same amount of measurements as the online version of experiment 6.
disp('Starting experiment 6 (offline)...');
tic;

% We calculate covariance matrices.
X = [Xu,Xs]; % This is the start of calculating the matrix covariances. We do NOT calculate Kmm, because this takes O(nm^2) time, which is what we want to prevent.
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.
K = lf^2*exp(-1/2*sum(diff.^2./repmat(permute(lx.^2,[2,3,1]),[n,n,1]),3)); % This is the covariance matrix. It contains the covariances of each combination of points.
KDivided = mat2cell(K,[nu,ns],[nu,ns]);
Kuu = KDivided{1,1};
Kus = KDivided{1,2};
Ksu = KDivided{2,1};
Kss = KDivided{2,2};
Kum = lf^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm6]) - repmat(permute(Xm(:,1:nm6),[3,2,1]),[nu,1])).^2./repmat(permute(lx.^2,[2,3,1]),[nu,nm6,1]),3));
Kmu = Kum';
Kmm = lf^2*ones(nm6,1); % We only take the diagonal elements of the original Kmm matrix here, because these are the only ones which we will need. We do not need the rest.
Qmm = sum((Kmu/Kuu).*Kmu,2); % This is the diagonal of Kmu/Kuu*Kum, but then calculated in a way that takes O(nm*nu^2) time instead of O(nm^2*nu) time.
Lmm = Kmm - Qmm; % This is \Lambda_{mm}. Or at least, its diagonal elements stored in a vector. There's no use storing the full matrix when the matrix is diagonal anyway.
Sfm = sfm^2*ones(nm6,1);
mm = zeros(nm6,1);
mu = zeros(nu,1);
ms = zeros(ns,1);

% We apply the offline FITC equations.
KumDivLmm = Kum./repmat((Lmm + Sfm)',nu,1); % We store this parameter because we need it multiple times. We use this method of calculating because it prevents us from having an nm by nm matrix.
Duu = Kuu + KumDivLmm*Kmu;
KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
Suu = KuuDivDuu*Kuu;
muu = KuuDivDuu*KumDivLmm*(fmh(1:nm6) - mm);
t6Offline = toc; % Having finished the training stage counts as being done.
mPost = ms + Ksu/Kuu*(muu - mu);
SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
RMSE6 = sqrt(sum(sum((mPost - fs).^2))/ns);
disp(['The FITC regression algorithm used ',num2str(nm6),' measurements in ',num2str(t6Offline),' seconds. The RMSE was ',num2str(RMSE6),'.']);