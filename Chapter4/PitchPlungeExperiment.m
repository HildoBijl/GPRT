% In this file we continue with the identification of the pitch-plunge system. There are (other than this initializing part zero) three parts in this script. The first part generates measurement
% data. It calls the Simulink file of the pitch-plunge system. Alternatively, if you don't like waiting, you can skip this part and just load in the data that I have already generated. The second
% part uses the data to generate plots of predictions. The third part then compares the runtime and accuracy of the algorithms with respect to the number of measurements used. After calling this
% first preamble part, you can run any part you like. Do make sure you load in the right data set at the start of part 2 and/or 3 though.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?

% We add paths containing files which we will need.
addpath('../PitchPlunge/Definitions/');
addpath('../PitchPlunge/Controllers/');
addpath('../ExportFig/');
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

%% Part one generates measurement data.

% We make some settings for the measurements that we will do.
nm = 100; % We set the number of time steps we want to feed to the GP. Note that generating measurements takes a lot of time, since we're using Simulink. We only have about 6 measurements per second.
rng(1, 'twister'); % We fix Matlab's random number generator, so that it always creates functions which I've found to be pretty representative as far as random samples go.
Xm = zeros(5,nm); % This set will contain all input data.
fmh = zeros(nm,4); % This set will contain all output data.

% To start off, we define timing data.
dt = 0.001; % We define the simulation time step.
T = 0.1; % We define the simulation length.
numDataPoints = ceil(T/dt)+1; % We calculate the number of data points we'll be having.

% We also set up other parameters. These are set to default values.
defineFlow; % We set up the flow properties.
defineStructuralParameters; % We define the structural parameters.
defineInitialConditions; % We define the initial conditions.
defineControllerParameters; % We set up the controller parameters.

% We define the range for the initial state.
h0Range = 5e-3;
a0Range = 6e-2;
hd0Range = 5e-2;
ad0Range = 1e0;
betaRange = 5e-1;
useNonlinear = 1; % We indicate we use the linear model for now.

% We define which controller we will use.
controller = @constantController;
global constantControllerValue;
constantControllerValue = 0;

% We loop through the experiments to obtain data.
for experiment = 1:nm
	% We set the initial state for the simulation.
	h0 = (rand(1,1)*2-1)*h0Range; % Initial plunge. [m]
	a0 = (rand(1,1)*2-1)*a0Range; % Initial pitch angle. [rad]
	hd0 = (rand(1,1)*2-1)*hd0Range; % Initial plunge rate. [m/s]
	ad0 = (rand(1,1)*2-1)*ad0Range; % Initial pitch angle rate. [rad/s]
	constantControllerValue = (rand(1,1)*2-1)*betaRange; % Control input. [rad]

	% We run the simulation.
	applySettings; % We apply the settings which have been set so far, also taking into account any possible adjustments that have been made. This also sets up the initial conditions.
	t = sim('../PitchPlunge/PitchPlunge');
	Xm(:,experiment) = [x(1,:)';xd(1,:)';constantControllerValue];
	fmh(experiment,:) = [x(end,:),xd(end,:)];
end

% We save the measurement data.
save('MeasurementData');

%% Part two processes the measurements and applies GP regression.

% We load the measurement data.
load('MeasurementData10k'); % This is the regular file with 10.000 measurements.
% load('MeasurementData10kFromStationary'); % In this file we only put the system in stationary states (\dot{h} = \dot{\alpha} = 0) and did not apply any input (beta = 0). As such, its approximations of this specific case will be much better, but its approximations of other cases would be worse. This file was used to generate the "true" output.
% load('MeasurementData'); % This is the measurement file which you just generated through the first part of this script.

% How many measurements should we use?
nm = 1000; % The number of measurements used. (GP is limited to 1000 by default.)
nmGP = min(nm, 1000); % We limit the number of measurements for GP to 1000, because otherwise things become too slow.

% We set up the inducing input points that we will use.
nuPerDimension = 9; % This is the number of inducing input points with respect to each dimension.
nu = nuPerDimension^2;
groupSize1 = 50; % What is the group size used within the PITC algorithm?
groupSize2 = 200; % What is the group size used within the PITC algorithm?
[xu1Mesh,xu2Mesh] = meshgrid(linspace(hMin,hMax,nuPerDimension),linspace(aMin,aMax,nuPerDimension));
Xu = [reshape(xu1Mesh,1,nu); reshape(xu2Mesh,1,nu)];
Xu = [Xu;zeros(3,nu)];

% We set up trial points, used for making plots.
hMin = -h0Range;
hMax = h0Range;
aMin = -a0Range;
aMax = a0Range;
nsPerDimension = 21; % This is the number of trial points per dimension.
ns = nsPerDimension^2; % This is the total number of trial points.
[x1Mesh,x2Mesh] = meshgrid(linspace(hMin,hMax,nsPerDimension),linspace(aMin,aMax,nsPerDimension));
Xs = [reshape(x1Mesh,1,ns); reshape(x2Mesh,1,ns)];
Xs = [Xs;zeros(3,ns)];

% We set the hyperparameters.
xScale = [h0Range;a0Range;hd0Range;ad0Range;betaRange];
sfm = xScale(1:4)'/100;
lf = xScale(1:4)*4;
lx = [xScale,xScale];
mb = zeros(4,1);

% We apply GP regression. First we set up test parameters.
tic;
GPFITC = zeros(2,1);

% We set up difference matrices.
X = [Xm(:,1:nmGP),Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We apply GP regression for each individual output.
for outputIndex = 1:2
	% We set up covariance matrices and such.
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
	KDivided = mat2cell(K,[nmGP,ns],[nmGP,ns]);
	Kmm = KDivided{1,1};
	Kms = KDivided{1,2};
	Ksm = KDivided{2,1};
	Kss = KDivided{2,2};
	Sfm = sfm(outputIndex)^2*eye(nmGP); % This is the noise covariance matrix.
	mm = mb(outputIndex)*ones(nmGP,1); % This is the mean vector m(Xm). We assume a constant mean function.
	ms = mb(outputIndex)*ones(ns,1); % This is the mean vector m(Xs). We assume a constant mean function.
	mPost = ms + Ksm/(Kmm + Sfm)*(fmh(1:nmGP,outputIndex) - mm); % This is the posterior mean vector.
	SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	
	% And then we plot the result.
	figure(0+outputIndex);
	clf(0+outputIndex);
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
		set(sDown,'FaceColor',[0.5,0.5,0.5]);
		set(sUp,'FaceColor',[0.5,0.5,0.5]);
		set(sMid,'FaceColor',[0.5,0.5,0.5]);
	else
		set(sDown,'FaceColor',[0,0,1]);
		set(sUp,'FaceColor',[0,0,1]);
		set(sMid,'FaceColor',[0,0,1]);
	end
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1.5e-2,1e-2]);
	end
	if outputIndex == 2
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-2e-1,2e-1]);
	end
	
	% We also plot the exact results.
% 	load('../Chapter2/CH2Predictions'); % These are the predictions from chapter 2, with a limited number of measurements.
	load('../Chapter4/CorrectPlot'); % These values were found by taking 10.000 experiments, which started from a stationary state and had zero input, and applying regular GP regression from it. It's the best we can do, so we assume that these are the true values.
	sPrevious = surface(x1Mesh, x2Mesh, mPostStorage(:,:,outputIndex));
	set(sPrevious,'FaceAlpha',0.4);
	if useColor == 0
		set(sPrevious,'FaceColor',[0.7,0.7,0.7]);
	else
		set(sPrevious,'FaceColor',[0.3,0.1,0]);
	end
	if exportFigs ~= 0
		export_fig(['NextStatePredictionFiveDimensional',num2str(outputIndex),'GP.png'],'-transparent');
	end
	
	% And we calculate the RMSE.
	RMSEGP(outputIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
end
tGP = toc;

% We apply FITC regression. First we set up test parameters.
tic;
RMSEFITC = zeros(2,1);

% We set up difference matrices.
X = [Xu,Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We apply GP regression for each individual output.
for outputIndex = 1:2
	% We set up covariance matrices and such.
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
	KDivided = mat2cell(K,[nu,ns],[nu,ns]);
	Kuu = KDivided{1,1};
	Kus = KDivided{1,2};
	Ksu = KDivided{2,1};
	Kss = KDivided{2,2};
	Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
	Kmu = Kum';
	Kmm = lf(outputIndex)^2*ones(nm,1); % We only take the diagonal elements of the original Kmm matrix here, because these are the only ones which we will need. We do not need the rest.
	Qmm = sum((Kmu/Kuu).*Kmu,2); % This is the diagonal of Kmu/Kuu*Kum, but then calculated in a way that takes O(nm*nu^2) time instead of O(nm^2*nu) time.
	Lmm = Kmm - Qmm; % This is \Lambda_{mm}. Or at least, its diagonal elements stored in a vector. There's no use storing the full matrix when the matrix is diagonal anyway.
	mm = mb(outputIndex)*ones(nm,1); % This is the prior mean vector of the measurement points.
	mu = mb(outputIndex)*ones(nu,1); % This is the prior mean vector of the inducing input points.
	ms = mb(outputIndex)*ones(ns,1); % This is the prior mean vector of the trial points.
	Sfm = sfm(outputIndex)^2*ones(nm,1); % This is the noise covariance matrix.
	% We apply the offline FITC equations.
	KumDivLmm = Kum./repmat((Lmm + Sfm)',nu,1); % We store this parameter because we need it multiple times. We use this method of calculating because it prevents us from having an nm by nm matrix.
	Duu = Kuu + KumDivLmm*Kmu;
	KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
	Suu = KuuDivDuu*Kuu;
	muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
	KsuDivKuu = Ksu/Kuu;
	mPost = ms + KsuDivKuu*(muu - mu);
	SPost = Kss - KsuDivKuu*(Kuu - Suu)*KsuDivKuu';
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
		
	% And then we plot the result.
	figure(2+outputIndex);
	clf(2+outputIndex);
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
		set(sDown,'FaceColor',[0.5,0.5,0.5]);
		set(sUp,'FaceColor',[0.5,0.5,0.5]);
		set(sMid,'FaceColor',[0.5,0.5,0.5]);
	else
		set(sDown,'FaceColor',[0,0,1]);
		set(sUp,'FaceColor',[0,0,1]);
		set(sMid,'FaceColor',[0,0,1]);
	end
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1.5e-2,1e-2]);
	end
	if outputIndex == 2
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-2e-1,2e-1]);
	end
	
	% We also plot the exact results.
% 	load('../Chapter2/CH2Predictions'); % These are the predictions from chapter 2, with a limited number of measurements.
	load('../Chapter4/CorrectPlot'); % These values were found by taking 10.000 experiments, which started from a stationary state and had zero input, and applying regular GP regression from it. It's the best we can do, so we assume that these are the true values.
	sPrevious = surface(x1Mesh, x2Mesh, mPostStorage(:,:,outputIndex));
	set(sPrevious,'FaceAlpha',0.4);
	if useColor == 0
		set(sPrevious,'FaceColor',[0.7,0.7,0.7]);
	else
		set(sPrevious,'FaceColor',[0.3,0.1,0]);
	end
	if exportFigs ~= 0
		export_fig(['NextStatePredictionFiveDimensional',num2str(outputIndex),'FITC.png'],'-transparent');
	end
	
	% And we calculate the RMSE.
	RMSEFITC(outputIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
end
tFITC = toc;

% We apply PITC regression. First we set up test parameters.
tic;
groupSize = groupSize1;
RMSEPITC1 = zeros(2,1);

% We set up difference matrices.
X = [Xu,Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We define the groups which we will use.
groupSizes = groupSize*ones(1,ceil(nm/groupSize)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
groupSizes(end) = mod(nm-1, groupSize)+1;
groups = mat2cell(1:nm, 1, groupSizes);

% We apply GP regression for each individual output.
for outputIndex = 1:2
	% We set up covariance matrices.
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
	KDivided = mat2cell(K,[nu,ns],[nu,ns]);
	Kuu = KDivided{1,1};
	Kus = KDivided{1,2};
	Ksu = KDivided{2,1};
	Kss = KDivided{2,2};
	Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
	Kmu = Kum';
	Sfm = sfm(outputIndex)^2*ones(nm,1);
	mm = zeros(nm,1);
	mu = zeros(nu,1);
	ms = zeros(ns,1);

	% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop.
	KumDivLmm = zeros(nu,nm); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
	for i = 1:length(groups)
		indices = groups{i}; % These are the indices of the measurements we will use in this group.
		Kmm = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
		Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
		KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
	end

	% Finally we apply the offline PITC equations.
	Duu = Kuu + KumDivLmm*Kmu;
	KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
	Suu = KuuDivDuu*Kuu;
	muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
	mPost = ms + Ksu/Kuu*(muu - mu);
	SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

	% And then we plot the result.
	figure(4+outputIndex);
	clf(4+outputIndex);
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
		set(sDown,'FaceColor',[0.5,0.5,0.5]);
		set(sUp,'FaceColor',[0.5,0.5,0.5]);
		set(sMid,'FaceColor',[0.5,0.5,0.5]);
	else
		set(sDown,'FaceColor',[0,0,1]);
		set(sUp,'FaceColor',[0,0,1]);
		set(sMid,'FaceColor',[0,0,1]);
	end
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1.5e-2,1e-2]);
	end
	if outputIndex == 2
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-2e-1,2e-1]);
	end
	
	% We also plot the exact results.
% 	load('../Chapter2/CH2Predictions'); % These are the predictions from chapter 2, with a limited number of measurements.
	load('../Chapter4/CorrectPlot'); % These values were found by taking 10.000 experiments, which started from a stationary state and had zero input, and applying regular GP regression from it. It's the best we can do, so we assume that these are the true values.
	sPrevious = surface(x1Mesh, x2Mesh, mPostStorage(:,:,outputIndex));
	set(sPrevious,'FaceAlpha',0.4);
	if useColor == 0
		set(sPrevious,'FaceColor',[0.7,0.7,0.7]);
	else
		set(sPrevious,'FaceColor',[0.3,0.1,0]);
	end
	if exportFigs ~= 0
		export_fig(['NextStatePredictionFiveDimensional',num2str(outputIndex),'PITC',num2str(groupSize1),'.png'],'-transparent');
	end
	
	% And we calculate the RMSE.
	RMSEPITC1(outputIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
end
tPITC1 = toc;

% We apply PITC regression again. First we set up test parameters.
tic;
groupSize = groupSize2;
RMSEPITC2 = zeros(2,1);

% We set up difference matrices.
X = [Xu,Xs];
n = size(X,2);
diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

% We define the groups which we will use.
groupSizes = groupSize*ones(1,ceil(nm/groupSize)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
groupSizes(end) = mod(nm-1, groupSize)+1;
groups = mat2cell(1:nm, 1, groupSizes);

% We apply GP regression for each individual output.
for outputIndex = 1:2
	% We set up covariance matrices.
	K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
	KDivided = mat2cell(K,[nu,ns],[nu,ns]);
	Kuu = KDivided{1,1};
	Kus = KDivided{1,2};
	Ksu = KDivided{2,1};
	Kss = KDivided{2,2};
	Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
	Kmu = Kum';
	Sfm = sfm(outputIndex)^2*ones(nm,1);
	mm = zeros(nm,1);
	mu = zeros(nu,1);
	ms = zeros(ns,1);

	% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop.
	KumDivLmm = zeros(nu,nm); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
	for i = 1:length(groups)
		indices = groups{i}; % These are the indices of the measurements we will use in this group.
		Kmm = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
		Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
		KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
	end

	% Finally we apply the offline PITC equations.
	Duu = Kuu + KumDivLmm*Kmu;
	KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
	Suu = KuuDivDuu*Kuu;
	muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
	mPost = ms + Ksu/Kuu*(muu - mu);
	SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
	sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
	mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
	sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

	% And then we plot the result.
	figure(6+outputIndex);
	clf(6+outputIndex);
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
		set(sDown,'FaceColor',[0.5,0.5,0.5]);
		set(sUp,'FaceColor',[0.5,0.5,0.5]);
		set(sMid,'FaceColor',[0.5,0.5,0.5]);
	else
		set(sDown,'FaceColor',[0,0,1]);
		set(sUp,'FaceColor',[0,0,1]);
		set(sMid,'FaceColor',[0,0,1]);
	end
	xlabel('h_k');
	ylabel('\alpha_k');
	if outputIndex == 1
		zlabel('h_{k+1}');
	else
		zlabel('\alpha_{k+1}');
	end
	if outputIndex == 1
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-1.5e-2,1e-2]);
	end
	if outputIndex == 2
		view([50,16])
		axis([-h0Range,h0Range,-a0Range,a0Range,-2e-1,2e-1]);
	end
	
	% We also plot the exact results.
% 	load('../Chapter2/CH2Predictions'); % These are the predictions from chapter 2, with a limited number of measurements.
	load('../Chapter4/CorrectPlot'); % These values were found by taking 10.000 experiments, which started from a stationary state and had zero input, and applying regular GP regression from it. It's the best we can do, so we assume that these are the true values.
	sPrevious = surface(x1Mesh, x2Mesh, mPostStorage(:,:,outputIndex));
	set(sPrevious,'FaceAlpha',0.4);
	if useColor == 0
		set(sPrevious,'FaceColor',[0.7,0.7,0.7]);
	else
		set(sPrevious,'FaceColor',[0.3,0.1,0]);
	end
	if exportFigs ~= 0
		export_fig(['NextStatePredictionFiveDimensional',num2str(outputIndex),'PITC',num2str(groupSize2),'.png'],'-transparent');
	end
	
	% And we calculate the RMSE.
	RMSEPITC2(outputIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
end
tPITC2 = toc;

% We display the final results.
disp(['The GP regression algorithm used ',num2str(nmGP),' measurements in ',num2str(tGP),' seconds. The RMSEs were ',num2str(RMSEGP(1)),' and ',num2str(RMSEGP(2)),'.']);
disp(['The PITC (',num2str(groupSize2),') regression algorithm used ',num2str(nm),' measurements in ',num2str(tPITC2),' seconds. The RMSEs were ',num2str(RMSEPITC2(1)),' and ',num2str(RMSEPITC2(2)),'.']);
disp(['The PITC (',num2str(groupSize1),') regression algorithm used ',num2str(nm),' measurements in ',num2str(tPITC1),' seconds. The RMSEs were ',num2str(RMSEPITC1(1)),' and ',num2str(RMSEPITC1(2)),'.']);
disp(['The FITC regression algorithm used ',num2str(nm),' measurements in ',num2str(tFITC),' seconds. The RMSEs were ',num2str(RMSEFITC(1)),' and ',num2str(RMSEFITC(2)),'.']);

%% Part three compares the accuracy and efficiency of the various algorithms for a varying number of measurements. This block may take a few minutes. It also may give warning for badly conditioned or singular matrices. These can mostly be ignored.

% We load the measurement data.
load('MeasurementData10k'); % This is the regular file with 10.000 measurements.
% load('MeasurementData10kFromStationary'); % In this file we only put the system in stationary states (\dot{h} = \dot{\alpha} = 0) and did not apply any input (beta = 0). As such, its approximations of this specific case will be much better, but its approximations of other cases would be worse. This file was used to generate the "true" output.
% load('MeasurementData'); % This is the measurement file which you just generated through the first part of this script.

% We define how many measurements we will use.
numExperiments = 3*5+1; % This is the number of full experiments which we will run for each algorithm. If nmMax is set to 10.000, then every experiment on average lasts about twenty seconds. Plan accordingly.
nmMin = 10; % This is the value of nm we will start with.
nmMax = 10000; % This is the value of nm we will end with. Note that in total we only have 10.000 measurements.
nmExp = linspace(log10(nmMin),log10(nmMax),numExperiments); % We use a logarithmic scale for the number of measurements used.
nmOptions = round(10.^nmExp);
groupSize1 = 50; % This is the group size we will use for the first PITC run.
groupSize2 = 200; % This is the group size we will use for the second PITC run.

% We set up storage parameters for the results.
numExperiments = length(nmOptions);
RMSEGP = zeros(2,numExperiments);
RMSEFITC = zeros(2,numExperiments);
RMSEPITC1 = zeros(2,numExperiments);
RMSEPITC2 = zeros(2,numExperiments);
tGP = zeros(1,numExperiments);
tFITC = zeros(1,numExperiments);
tPITC1 = zeros(1,numExperiments);
tPITC2 = zeros(1,numExperiments);

% We browse through the number of measurements options.
for nmIndex = 1:length(nmOptions)
	% We extract nm.
	nm = nmOptions(nmIndex);
	nmGP = nm;
	disp(['We are currently at experiment run ',num2str(nmIndex),' out of ',num2str(numExperiments),'. The number of measurements used in it is ',num2str(nm),'.']);

	% We apply GP regression. We set up difference matrices.
	tic;
	X = [Xm(:,1:nmGP),Xs];
	n = size(X,2);
	diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

	% We apply GP regression for each individual output.
	for outputIndex = 1:2
		% We set up covariance matrices and such.
		K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
		KDivided = mat2cell(K,[nmGP,ns],[nmGP,ns]);
		Kmm = KDivided{1,1};
		Kms = KDivided{1,2};
		Ksm = KDivided{2,1};
		Kss = KDivided{2,2};
		Sfm = sfm(outputIndex)^2*eye(nmGP); % This is the noise covariance matrix.
		mm = mb(outputIndex)*ones(nmGP,1); % This is the mean vector m(Xm). We assume a constant mean function.
		ms = mb(outputIndex)*ones(ns,1); % This is the mean vector m(Xs). We assume a constant mean function.
		mPost = ms + Ksm/(Kmm + Sfm)*(fmh(1:nmGP,outputIndex) - mm); % This is the posterior mean vector.
		SPost = Kss - Ksm/(Kmm + Sfm)*Kms; % This is the posterior covariance matrix.
		sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
		mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
		sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

		% And we calculate the RMSE.
		RMSEGP(outputIndex,nmIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
	end
	tGP(nmIndex) = toc;

	% We apply FITC regression. We set up difference matrices.
	tic;
	X = [Xu,Xs];
	n = size(X,2);
	diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

	% We apply GP regression for each individual output.
	for outputIndex = 1:2
		% We set up covariance matrices and such.
		K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
		KDivided = mat2cell(K,[nu,ns],[nu,ns]);
		Kuu = KDivided{1,1};
		Kus = KDivided{1,2};
		Ksu = KDivided{2,1};
		Kss = KDivided{2,2};
		Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
		Kmu = Kum';
		Kmm = lf(outputIndex)^2*ones(nm,1); % We only take the diagonal elements of the original Kmm matrix here, because these are the only ones which we will need. We do not need the rest.
		Qmm = sum((Kmu/Kuu).*Kmu,2); % This is the diagonal of Kmu/Kuu*Kum, but then calculated in a way that takes O(nm*nu^2) time instead of O(nm^2*nu) time.
		Lmm = Kmm - Qmm; % This is \Lambda_{mm}. Or at least, its diagonal elements stored in a vector. There's no use storing the full matrix when the matrix is diagonal anyway.
		mm = mb(outputIndex)*ones(nm,1); % This is the prior mean vector of the measurement points.
		mu = mb(outputIndex)*ones(nu,1); % This is the prior mean vector of the inducing input points.
		ms = mb(outputIndex)*ones(ns,1); % This is the prior mean vector of the trial points.
		Sfm = sfm(outputIndex)^2*ones(nm,1); % This is the noise covariance matrix.
		
		% We apply the offline FITC equations.
		KumDivLmm = Kum./repmat((Lmm + Sfm)',nu,1); % We store this parameter because we need it multiple times. We use this method of calculating because it prevents us from having an nm by nm matrix.
		Duu = Kuu + KumDivLmm*Kmu;
		KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
		Suu = KuuDivDuu*Kuu;
		muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
		KsuDivKuu = Ksu/Kuu;
		mPost = ms + KsuDivKuu*(muu - mu);
		SPost = Kss - KsuDivKuu*(Kuu - Suu)*KsuDivKuu';
		sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
		mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
		sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

		% And we calculate the RMSE.
		RMSEFITC(outputIndex,nmIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
	end
	tFITC(nmIndex) = toc;

	% We apply PITC regression. We set up difference matrices.
	tic;
	X = [Xu,Xs];
	n = size(X,2);
	diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

	% We define the groups which we will use.
	groupSizes = groupSize1*ones(1,ceil(nm/groupSize1)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
	groupSizes(end) = mod(nm-1, groupSize1)+1;
	groups = mat2cell(1:nm, 1, groupSizes);

	% We apply GP regression for each individual output.
	for outputIndex = 1:2
		% We set up covariance matrices.
		K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
		KDivided = mat2cell(K,[nu,ns],[nu,ns]);
		Kuu = KDivided{1,1};
		Kus = KDivided{1,2};
		Ksu = KDivided{2,1};
		Kss = KDivided{2,2};
		Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
		Kmu = Kum';
		Sfm = sfm(outputIndex)^2*ones(nm,1);
		mm = zeros(nm,1);
		mu = zeros(nu,1);
		ms = zeros(ns,1);

		% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop.
		KumDivLmm = zeros(nu,nm); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
		for i = 1:length(groups)
			indices = groups{i}; % These are the indices of the measurements we will use in this group.
			Kmm = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
			Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
			KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
		end

		% Finally we apply the offline PITC equations.
		Duu = Kuu + KumDivLmm*Kmu;
		KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
		Suu = KuuDivDuu*Kuu;
		muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
		mPost = ms + Ksu/Kuu*(muu - mu);
		SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
		sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
		mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
		sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

		% And we calculate the RMSE.
		RMSEPITC1(outputIndex,nmIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
	end
	tPITC1(nmIndex) = toc;

	% We apply PITC regression. We set up difference matrices.
	tic;
	X = [Xu,Xs];
	n = size(X,2);
	diff = repmat(permute(X,[2,3,1]),[1,n]) - repmat(permute(X,[3,2,1]),[n,1]); % This is matrix containing differences between input points.

	% We define the groups which we will use.
	groupSizes = groupSize2*ones(1,ceil(nm/groupSize2)); % Here we set up the groups. We first define the sizes of each group and then give each group the corresponding indices.
	groupSizes(end) = mod(nm-1, groupSize2)+1;
	groups = mat2cell(1:nm, 1, groupSizes);

	% We apply GP regression for each individual output.
	for outputIndex = 1:2
		% We set up covariance matrices.
		K = lf(outputIndex)^2*exp(-1/2*sum(diff.^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[n,n,1]),3));
		KDivided = mat2cell(K,[nu,ns],[nu,ns]);
		Kuu = KDivided{1,1};
		Kus = KDivided{1,2};
		Ksu = KDivided{2,1};
		Kss = KDivided{2,2};
		Kum = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xu,[2,3,1]),[1,nm]) - repmat(permute(Xm(:,1:nm),[3,2,1]),[nu,1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[nu,nm,1]),3));
		Kmu = Kum';
		Sfm = sfm(outputIndex)^2*ones(nm,1);
		mm = zeros(nm,1);
		mu = zeros(nu,1);
		ms = zeros(ns,1);

		% We now set up covariance matrices differing per group. I couldn't find a way to do this without a loop, so we use a loop.
		KumDivLmm = zeros(nu,nm); % This will equal Kum/(\Lambda_{mm} + \hat{\Sigma}_{f_m})^{-1}.
		for i = 1:length(groups)
			indices = groups{i}; % These are the indices of the measurements we will use in this group.
			Kmm = lf(outputIndex)^2*exp(-1/2*sum((repmat(permute(Xm(:,indices),[2,3,1]),[1,length(indices)]) - repmat(permute(Xm(:,indices),[3,2,1]),[length(indices),1])).^2./repmat(permute(lx(:,outputIndex).^2,[2,3,1]),[length(indices),length(indices),1]),3)); % This is the matrix Kmm of the current group.
			Lmm = Kmm - Kmu(indices,:)/Kuu*Kum(:,indices);
			KumDivLmm(:,indices) = Kum(:,indices)/(Lmm + diag(Sfm(indices)));
		end

		% Finally we apply the offline PITC equations.
		Duu = Kuu + KumDivLmm*Kmu;
		KuuDivDuu = Kuu/Duu; % We store this parameter because we need it multiple times.
		Suu = KuuDivDuu*Kuu;
		muu = KuuDivDuu*KumDivLmm*(fmh(1:nm,outputIndex) - mm);
		mPost = ms + Ksu/Kuu*(muu - mu);
		SPost = Kss - Ksu/Kuu*(Kuu - Suu)/Kuu*Kus;
		sPost = sqrt(diag(SPost)); % These are the posterior standard deviations of the predictions of the trial function values.
		mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
		sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

		% And we calculate the RMSE.
		RMSEPITC2(outputIndex,nmIndex) = sqrt(sum(sum((mPost - mPostStorage(:,:,outputIndex)).^2))/ns);
	end
	tPITC2(nmIndex) = toc;
end

% Finally, we plot the results of the experiments.

% Optionally, we can load in the data of a long run of experiments. For this, we do need to call the first block of this script first, setting settings such as whether or not to export figures.
% load('PitchPlungeExperimentData');

figure(9);
clf(9);
hold on;
grid on;
xlabel('Number of measurements used');
ylabel('Runtime required');
plot(nmOptions,tGP,'-','Color',blue);
plot(nmOptions,tPITC2,'-','Color',red);
plot(nmOptions,tPITC1,'-','Color',green);
plot(nmOptions,tFITC,'-','Color',yellow);
axis([0,max(nmOptions),0,3]);
legend('GP',['PITC (',num2str(groupSize2),')'],['PITC (',num2str(groupSize1),')'],'FITC');
if exportFigs ~= 0
	export_fig('RuntimeVersusNumMeasurements.png','-transparent');
end

figure(10);
clf(10);
hold on;
grid on;
xlabel('Number of measurements used');
ylabel('RMSE on output h_{k+1}');
plot(nmOptions,RMSEGP(1,:),'-','Color',blue);
plot(nmOptions,RMSEPITC2(1,:),'-','Color',red);
plot(nmOptions,RMSEPITC1(1,:),'-','Color',green);
plot(nmOptions,RMSEFITC(1,:),'-','Color',yellow);
axis([0,max(nmOptions),0,1.2e-3]);
legend('GP',['PITC (',num2str(groupSize2),')'],['PITC (',num2str(groupSize1),')'],'FITC');
if exportFigs ~= 0
	export_fig('RMSEOutput1VersusNumMeasurements.png','-transparent');
end

figure(11);
clf(11);
hold on;
grid on;
xlabel('Number of measurements used');
ylabel('RMSE on output \alpha_{k+1}');
plot(nmOptions,RMSEGP(2,:),'-','Color',blue);
plot(nmOptions,RMSEPITC2(2,:),'-','Color',red);
plot(nmOptions,RMSEPITC1(2,:),'-','Color',green);
plot(nmOptions,RMSEFITC(2,:),'-','Color',yellow);
axis([0,max(nmOptions),0,10e-3]);
legend('GP',['PITC (',num2str(groupSize2),')'],['PITC (',num2str(groupSize1),')'],'FITC');
if exportFigs ~= 0
	export_fig('RMSEOutput2VersusNumMeasurements.png','-transparent');
end