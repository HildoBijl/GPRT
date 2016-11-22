% This file contains the experiment on executing the MCMD algorithm. With the predefined parameters, running this script should take four or five minutes. Of course if more measurements, particles
% or challenge rounds are added, the runtime will increase. If you only want to get the result plots, only run the first and third block and skip the middle block.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 0; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.
addpath('../Tools'); % We load in the acquisition functions.

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
	colormap(repmat((0:0.01:1)',1,3));
else
	red = [0.8 0 0];
	green = [0 0.4 0];
	blue = [0 0 0.8];
	yellow = [0.6 0.6 0];
	grey = [0.8 0.8 1];
	brown = [0.45 0.15 0.0];
	colormap('default');
end

% We define settings for the script.
nRuns = 1; % This is the number of full runs we do for every acquisition function. In the end we average over all the results.
nInputs = 50; % This is the number of try-out inputs every acquisition function can try during a full run.
displayProgress = 1; % Do we want to get updates from the script about how it's doing?
displayPlots = 1; % Do we want to get plots from the script results?
nStarts = 3; % When optimizing an acquisition function, this is the number of starts per dimension which we get in the multi-start optimization algorithm.
dx = 2; % This is the dimension of the input vector.
OptimizationFunction = @(x)(BraninFunction(x)); % This is the function we want to optimize.
nAF = 4; % How many acquisition functions are there?
xMin = [-5;0]; % This is the minimum input for each dimension.
xMax = [10;15]; % This is the maximum input for each dimension.
nsPerDimension = 31; % If we make plots, how many plot (trial) points do we use for each dimension?

% We set up a plot of the function.
ns = nsPerDimension^2; % This is the number of plot points in total.
x1Range = linspace(xMin(1),xMax(1),nsPerDimension); % These are the plot points in the dimension of x1.
x2Range = linspace(xMin(2),xMax(2),nsPerDimension); % These are the plot points in the dimension of x2.
[x1Mesh,x2Mesh] = meshgrid(x1Range,x2Range); % We turn the plot points into a grid.
Xs = [reshape(x1Mesh,1,ns);reshape(x2Mesh,1,ns)]; % These are the trial points used for generating plot data.
rng(2, 'twister'); % We fix Matlab's random number generator, so we get the same plots as found online. Although actually this does not seem to work this time. Though I have no idea why.
fs = zeros(ns,1); % These are the function values for the trial points.
for i = 1:ns
	fs(i) = OptimizationFunction(Xs(:,i));
end
fsMesh = reshape(fs,nsPerDimension,nsPerDimension);

% We actually make the plot.
figure(3);
clf(3);
hold on;
grid on;
meshc(x1Mesh,x2Mesh,fsMesh);
surface(x1Mesh,x2Mesh,fsMesh);
if useColor == 0
	colormap(((0:0.01:0.8)'*ones(1,3)).^.8)
else
	colormap('default');
end
xlabel('x_1');
ylabel('x_2');
zlabel('Branin function output f(x_1,x_2)');
view([20,25]);
if exportFigs ~= 0
	export_fig('BraninFunction.png','-transparent');
end

% We set up the GP properties.
lf = 250; % This is the output length scale.
lx = [4;18]; % This is the input length scale.
sfm = 5; % This is the standard deviation of the noise.
eps = 1e-10; % This is a small number we use for numerical reasons.

% We set up the MCMD settings.
np = 1e3; % We define the number of particles used.
alpha = 1/3; % This is the part of the time we sample a challenger from the current belief of the maximum distribution. A high value of alpha (near 1) speeds up convergence, but distorts the results slightly. For larger problems it is wise to start with a high value of alpha but decrease it towards zero as the algorithm converges. We will change it later on in the algorithm.
h = 0.2*ones(dx,1); % This is the length scale for each dimension of the Gaussian kernel we will use in the kernel density estimation process.

% We set up other acquisition function settings. These have been tuned manually.
kappa = 2; % This is the exploration parameter for the UCB function.
xiPI = 2; % This is the exploration parameter for the PI function.
xiEI = 2; % This is the exploration parameter for the EI function.

%% We set up storage parameters.
sXm = zeros(dx, nInputs, nAF, nRuns); % These are the chosen measurement points.
sf = zeros(nInputs, nAF, nRuns); % These are the function values at these measurement points.
sfh = zeros(nInputs, nAF, nRuns); % These are the measured function values (with measurement noise).
snu = zeros(nAF, nRuns); % This is a counter that keeps track of how many inducing input points we have.
sXu = zeros(dx, nInputs, nAF, nRuns); % These are the inducing input points that we will use.
smuu = zeros(nInputs, nAF, nRuns); % This is the mean vector of the inducing function values.
sSuu = zeros(nInputs, nInputs, nAF, nRuns); % This is the posterior covariance matrix of the inducing function values.
sKuu = zeros(nInputs, nInputs, nAF, nRuns); % This is the prior covariance matrix of the inducing function values.
sParticles = zeros(dx, np, nRuns); % These are the particles at each run.
sWeights = zeros(np, nRuns); % These are the particle weights at each run.
sRecommendations = zeros(dx, nInputs, nAF, nRuns); % These are the recommendations made at the end of all the measurements.
sRecommendationValue = zeros(nInputs, nAF, nRuns); % These are the values at the given recommendation points.
sRecommendationBelievedValue = zeros(nInputs, nAF, nRuns); % These are the values which the GP beliefs we would get at the recommendation points.

% We start doing the experiment runs.
tic;
for run = 1:nRuns
	% We keep track of how far along we are.
	if displayProgress >= 1
		disp(['Starting full experiment run ',num2str(run),'/',num2str(nRuns),'. Time passed is ',num2str(toc),' seconds.']);
	end
	
	% We select a first point to try. This is the same for all acquisition functions to reduce random effects. (If one AF gets a lucky first point and another a crappy one, it's kind of unfair.)
	x0 = rand(dx,1).*(xMax-xMin) + xMin; % This is the first input point used for all acquisition functions.
	f0 = OptimizationFunction(x0); % This is the corresponding function value.
	fh0 = f0 + sfm*randn(1,1); % This is the measured value (with noise) for the first input point.
	
	% We already set up the particles for the MCMD algorithm.
	sParticles(:, :, run) = rand(dx,np).*repmat(xMax-xMin,1,np) + repmat(xMin,1,np);
	sWeights(:, run) = ones(np,1);
	
	for iAF = 1:nAF
		% We keep track of how far along we are.
		if displayProgress >= 1
			disp(['Starting acquisition function ',num2str(iAF),'. Time passed is ',num2str(toc),' seconds.']);
		end
		
		% We implement the first measurement that we have already done.
		sXm(:, 1, iAF, run) = x0;
		sf(1, iAF, run) = f0;
		sfh(1, iAF, run) = fh0;
		
		% We set up the inducing input points and their function value distribution prior to incorporating the first measurement.
		nu = 1;
		Xu = x0;
		muu = 0;
		Kuu = lf^2;
		Suu = lf^2;
		
		% And then we incorporate the first measurement. We use sparse online GP regression here.
		Kpp = lf^2;
		Kpu = lf^2*exp(-1/2*sum((permute(Xu,[3,2,1]) - repmat(permute(x0,[2,3,1]),1,nu)).^2./repmat(permute(lx.^2,[3,2,1]),1,nu),3)); % This is the covariance matrix K+u.
		Kup = Kpu';
		Spp = Kpp - Kpu/(Kuu+eps*eye(size(Kuu)))*(Kuu - Suu)/(Kuu+eps*eye(size(Kuu)))*Kup;
		Spu = Kpu/(Kuu+eps*eye(size(Kuu)))*Suu;
		Sup = Spu';
		mup = Kpu/(Kuu+eps*eye(size(Kuu)))*muu;
		muu = muu + Sup/(Kpp + sfm^2)*(fh0 - mup);
		Suu = Suu - Sup/(Kpp + sfm^2)*Spu;
	
		% We calculate the first recommendation which would be given, based on the data so far.
		[xOpt, fOpt] = optimizeAcquisitionFunction(@(x)(AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)), xMin, xMax, nStarts);
		sRecommendations(:, 1, iAF, run) = xOpt;
		sRecommendationBelievedValue(1, iAF, run) = fOpt;
		sRecommendationValue(1, iAF, run) = OptimizationFunction(xOpt);
		
		% We now start iterating over all the input points. We start from the second point because the first is the same for all acquisition functions anyway. We have already incorporated it.
		for i = 2:nInputs
			% We find the next try-out point. How to do this depends on which method we are using.
			if iAF ~= 1 % Are we using a regular acquisition function?
				% So we are using a regular acquisition function. We first look at what is the best point which we can get in the first place. That is, the one with the highest mean.
				[~, fOpt] = optimizeAcquisitionFunction(@(x)(AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)), xMin, xMax, nStarts);

				% We choose an acquisition function to optimize.
				if iAF == 2
					AF = @(x)(AFUCB(x, Xu, muu, Suu, Kuu, lf, lx, eps, kappa)); % This is the upper confidence bound acquisition function.
				elseif iAF == 3
					AF = @(x)(AFPI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xiPI)); % This is the probability of improvement acquisition function.
				else
					AF = @(x)(AFEI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xiEI)); % This is the expected improvement acquisition function.
				end

				% We run a multi-start optimization on the acquisition function. 
				[xOpt, afMax] = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
				if (iAF == 3 || iAF == 4)
					if iAF == 3
						xi = xiPI;
					else
						xi = xiEI;
					end
					while afMax < -1e100 % For acquisition functions 4 and 5, we run an extra check, since due to numerical issues they sometimes find the wrong solution with an overly low acquisition function value.
						xi = xi/2; % The cause is a too high value of zeta, so we lower it.
						AF = @(x)(AFPI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xi)); % We implement the new value of zeta into the AF.
						[xOpt, afMax] = optimizeAcquisitionFunction(AF, xMin, xMax, nStarts);
						disp(['We could not find an optimum in the acquisition function at run ',num2str(run),', AF ',num2str(iAF),', measurement ',num2str(i),'. The acquisition function was too small.']);
					end
				end
			else % We are using the MCMD algorithm with Thompson sampling.
				% We start doing the challenge rounds.
				nr = max(2,ceil(10 - i/5)); % This is the number of rounds we use. We start with a relatively large number of rounds, but this number decreases, since we do not need so many rounds after a while anyway.
				for round = 1:nr
					% We start by applying systematic resampling. (Yes, this is quite useless in the first round, but we ignore that tiny detail and do it anyway.)
					oldParticles = sParticles(:, :, run); % We store the old particles, so we can override the particles matrix during the process of resampling.
					oldWeights = sWeights(:, run);
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
							sParticles(:, newPCounter, run) = oldParticles(:, oldPCounter);
							sWeights(newPCounter, run) = 1;
							newPCounter = newPCounter + 1;
						end
					end

					% We now create challengers according to the specified rules.
					alpha = 2/3 + (1/6-2/3)*(i/nInputs);
					sampleFromMaxDist = (rand(1,np) < alpha); % We determine which challengers we will pick from the current belief of the maximum distribution, and which challengers we pick randomly.
					randomPoints = rand(dx,np).*repmat(xMax-xMin,1,np)+repmat(xMin,1,np); % We select random challengers. (We do this for all particles and then discard the ones which we do not need.)
					indices = ceil(rand(np,1)*np); % We pick the indices of the champion particles we will use to generate challengers from. This line only works if all the particles have a weight of one, which is the case since we have just resampled. Otherwise, we should use randsample(np,np,true,weights);
					deviations = randn(dx,np).*repmat(h,1,np); % To apply the Gaussian kernel to the selected champions, we need to add a Gaussian parameter to the champion particles. We set that one up here.
					challengers = repmat(1-sampleFromMaxDist,dx,1).*randomPoints + repmat(sampleFromMaxDist,dx,1).*(sParticles(:, indices, run) + deviations); % We finalize the challenger points, picking random ones where applicable and sampled from the maximum distribution in other cases.

					% We now set up the covariance matrices and calculate some preliminary parameters.
					Kpu = lf^2*exp(-1/2*sum((repmat(permute(Xu,[3,2,1]),2*np,1) - repmat(permute([sParticles(:, :, run),challengers],[2,3,1]),1,nu)).^2./repmat(permute(lx.^2,[3,2,1]),2*np,nu),3)); % This is the covariance matrix between the inducing input points and all the selected champion/challenger points.
					KpuDivKuu = Kpu/(Kuu+eps*eye(size(Kuu)));
					mup = KpuDivKuu*muu; % These are the mean values of the Gaussian process at all the particle points.

					% We calculate the mean and covariance for each combination of challenger and challenged point. Then we sample \hat{f} and look at the result.
					oldParticles = sParticles(:, :, run); % We store the old particles, so we can override the particles matrix during the process of resampling.
					oldWeights = sWeights(:, run);
					for j = 1:np
						mupc = mup([j,j+np]); % This is the current mean vector.
						Kppc = lf^2*exp(-1/2*sum((repmat(permute([sParticles(:, j, run),challengers(:, j)],[3,2,1]),2,1) - repmat(permute([sParticles(:, j, run),challengers(:, j)],[2,3,1]),1,2)).^2./repmat(permute(lx.^2,[3,2,1]),2,2),3));
						Sigmac = Kppc - KpuDivKuu([j,j+np],:)*(Kuu - Suu)*KpuDivKuu([j,j+np],:)'; % This is the current covariance matrix.
						try % We use a try-catch-block here because sometimes numerical errors may occur.
							fHat = mupc + chol(Sigmac)'*randn(2,1);
							if fHat(2) > fHat(1) % Has the challenger won?
								sParticles(:, j, run) = challengers(:, j);
								q = 1/(xMax-xMin); % This is the probability density function value of q(x).
								qp = (1-alpha)*q + alpha*1/prod(sqrt(2*pi)*h)*exp(-1/2*sum((challengers(:, j) - oldParticles(:, indices(j))).^2./h.^2)); % This is the sampling probability density function given that we have selected the champion particle from the indices vector in the sampling process.
								sWeights(j, run) = q/qp;
							end
						catch
							% Apparently challengerPoints(i) and points(i) are so close together that we have numerical problems. Since they're so close, we can just ignore this case anyway, except possibly
							% display that numerical issues may have occurred.
							disp(['There may be numerical issues in the challenging process at particle ',num2str(j),'.']);
% 							keyboard; % I recommend checking Sigmac. When this is indeed singular, because sParticles(:, j, run) and challengers(:, j) are identical, just enter "return" to let the script continue.
						end
					end
				end
				
				% We are done updating the particle distribution. Finally we need to take a random sample from it. For this, we pick a random existing particle (taking into account weights) and
				% sample a new particle from its kernel function. We use the Gaussian kernel here.
				particleIndex = randsample(np, 1, true, sWeights(:, run));
				xOpt = sParticles(:, particleIndex, run) + randn(dx,1).*h; % We sample a particle from the kernel of the chosen champion particle.
			end % End of check on which acquisition function to use.
			
			% We store the selected try-out point, look up the function value and turn it into a measurement.
			sXm(:, i, iAF, run) = xOpt;
			sf(i, iAF, run) = OptimizationFunction(xOpt);
			sfh(i, iAF, run) = sf(i, iAF, run) + sfm*randn(1,1);

			% We calculate the prior distribution for this new point.
			Kpp = lf^2; % This is the prior covariance K++ of the new point x+.
			Kpu = lf^2*exp(-1/2*sum((permute(Xu,[3,2,1]) - repmat(permute(xOpt,[2,3,1]),1,nu)).^2./repmat(permute(lx.^2,[3,2,1]),1,nu),3)); % This is the covariance matrix K+u.
			Kup = Kpu';
			KpuDivKuu = Kpu/(Kuu+eps*eye(size(Kuu)));
			Spp = Kpp - KpuDivKuu*(Kuu - Suu)*KpuDivKuu'; % This is the posterior covariance \Sigma_{++} of the new point.
			Spu = KpuDivKuu*Suu;
			Sup = Spu';
			mup = KpuDivKuu*muu;

			% We check if we need to add an inducing input point or not. We do this by seeing if this point is close to any already existing inducing input point.
			ud = 3*(1/14)/(1/14 + i/nInputs); % This is the separation required for a new inducing input point. We decrease it as the algorithm progresses.
			if min(sum((Xu - repmat(xOpt,1,nu)).^2,1)) > ud^2
				% We update the inducing input distribution.
				nu = nu + 1;
				Xu = [Xu,xOpt];
				muu = [muu; mup];
				Kuu = [Kuu, Kup; Kpu, Kpp];
				Suu = [Suu, Sup; Spu, Spp];
				% We also update the covariance matrices of the new point with respect to the inducing input points, because this has changed too with the new inducing input point.
				Kpu = [Kpu, lf^2];
				Kup = [Kup; lf^2];
				Spu = [Spu, Spp];
				Sup = [Sup; Spp];
			end

			% And then we incorporate the measurement into the inducing function value distribution using sparse online GP regression.
			muu = muu + Sup/(Spp + sfm^2)*(sfh(i, iAF, run) - mup);
			Suu = Suu - Sup/(Spp + sfm^2)*Spu;
			
			% Now that the measurement has been incorporated, we let the algorithm make a recommendation of the input, based on all data so far. This is equal to the highest mean. We use this to
			% calculate the instantaneous regret.
			[xOpt, fOpt] = optimizeAcquisitionFunction(@(x)(AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)), xMin, xMax, nStarts);
			sRecommendations(:, i, iAF, run) = xOpt;
			sRecommendationBelievedValue(i, iAF, run) = fOpt;
			sRecommendationValue(i, iAF, run) = OptimizationFunction(xOpt);
		end
		
		% At the end we store the inducing input points and corresponding parameters, so we can use them to calculate stuff later on, if desired.
		snu(iAF, run) = nu;
		sXu(:, 1:nu, iAF, run) = Xu;
		smuu(1:nu, iAF, run) = muu;
		sKuu(1:nu, 1:nu, iAF, run) = Kuu;
		sSuu(1:nu, 1:nu, iAF, run) = Suu;
		
		% If desired, we also generate a plot of the result.
		if displayPlots ~= 0
			% We start by displaying the Gaussian process resulting from the measurements. We make the calculations for the trial points.
			X = [Xu,Xs];
			n = size(X,2);
			K = lf^2*exp(-1/2*sum((repmat(permute(X,[3,2,1]),n,1) - repmat(permute(X,[2,3,1]),1,n)).^2./repmat(permute(lx.^2,[3,2,1]),n,n), 3)); % This is the covariance matrix. It contains the covariances of each combination of points.
			KDivided = mat2cell(K,[nu,ns],[nu,ns]);
			Kuu = KDivided{1,1};
			Kus = KDivided{1,2};
			Ksu = KDivided{2,1};
			Kss = KDivided{2,2};
			ms = zeros(ns,1); % This is m(X_*).
			mu = zeros(nu,1); % This is m(X_u).
			mPost = ms + Ksu/(Kuu+eps*eye(size(Kuu)))*(muu - mu);
			SPost = Kss - Ksu/(Kuu+eps*eye(size(Kuu)))*(Kuu - Suu)/(Kuu+eps*eye(size(Kuu)))*Kus;
			sPost = sqrt(diag(SPost)); % These are the posterior standard deviations.
			su = sqrt(diag(Suu)); % These are the standard deviations of the inducing input points.
			mPost = reshape(mPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.
			sPost = reshape(sPost, nsPerDimension, nsPerDimension); % We put the result in a square format again.

			% We plot the resulting Gaussian process.
			figNum = (run-1)*(2*nAF+1) + iAF*2 + 2;
			figure(figNum);
			clf(figNum);
			hold on;
			grid on;
			xlabel('x_1');
			ylabel('x_2');
			if iAF == 1
				zlabel('MCMD output');
			elseif iAF == 2
				zlabel('UCB output');
			elseif iAF == 3
				zlabel('PI output');
			else
				zlabel('EI output');
			end
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
				scatter3(sXm(1,:,iAF,run), sXm(2,:,iAF,run), sf(:,iAF,run), 'ko', 'filled');
			else
				set(sDown,'FaceColor',[0,0,1]);
				set(sUp,'FaceColor',[0,0,1]);
				set(sMid,'FaceColor',[0,0,1]);
				scatter3(sXm(1,:,iAF,run), sXm(2,:,iAF,run), sf(:,iAF,run), 'ro', 'filled');
			end
			axis([xMin(1),xMax(1),xMin(2),xMax(2),-350,100]);
			view([20,25]);
			if exportFigs ~= 0
				if iAF == 1
					export_fig(['MCMDOutput2DRun',num2str(run),'.png'],'-transparent');
				elseif iAF == 2
					export_fig(['UCBOutput2DRun',num2str(run),'.png'],'-transparent');
				elseif iAF == 3
					export_fig(['PIOutput2DRun',num2str(run),'.png'],'-transparent');
				else
					export_fig(['EIOutput2DRun',num2str(run),'.png'],'-transparent');
				end
			end

			% We set up either the maximum distribution (for the MCMD algorithm) or the acquisition function (for the acquisition functions).
			figNum = (run-1)*(2*nAF+1) + iAF*2 + 3;
			figure(figNum);
			clf(figNum);
			hold on;
			grid on;
			xlabel('x_1');
			ylabel('x_2');
			if iAF == 1
				% We set up a storage parameter for the maximum distributions, and we set up the PDF for it.
				pMax = zeros(1,ns);
				for k = 1:np
					pMax = pMax + sWeights(k,run)*1/prod(sqrt(2*pi)*h)*exp(-1/2*sum((Xs - repmat(sParticles(:,k,run),1,ns)).^2./repmat(h.^2,1,ns),1));
				end
				pMax = pMax/sum(sWeights(:,run));
				pMax = reshape(pMax, nsPerDimension, nsPerDimension); % We put the result in a square format again.
				zlabel('Particle distribution');
				meshc(x1Mesh,x2Mesh,pMax);
				surface(x1Mesh,x2Mesh,pMax);
				if useColor == 0
					colormap(((0:0.01:0.8)'*ones(1,3)).^.8)
				else
					colormap('default');
				end
				view([20,25]);
				if exportFigs ~= 0
					export_fig(['MCMDMaximumDistribution2DRun',num2str(run),'.png'],'-transparent');
				end
				
				% We also set up a figure for the true maximum distribution of the posterior GP. First we calculate this true maximum distribution.
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
				limitDist = limitDist/prod((xMax - xMin)/(nsPerDimension - 1)); % We turn the result into a PDF.
				limitDist = reshape(limitDist, nsPerDimension, nsPerDimension); % We put the result in a square format again.
				
				% And now we plot the true maximum distribution.
				figNum = (run-1)*(2*nAF+1) + iAF*2 + 1;
				figure(figNum);
				clf(figNum);
				hold on;
				grid on;
				xlabel('x_1');
				ylabel('x_2');
				zlabel('Maximum distribution');
				meshc(x1Mesh,x2Mesh,limitDist);
				surface(x1Mesh,x2Mesh,limitDist);
				if useColor == 0
					colormap(((0:0.01:0.8)'*ones(1,3)).^.8)
				else
					colormap('default');
				end
				view([20,25]);
				if exportFigs ~= 0
					export_fig(['MaximumDistribution2DRun',num2str(run),'.png'],'-transparent');
				end
			else
				% We check which acquisition function we are using.
				if iAF == 2
					AF = @(x)(AFUCB(x, Xu, muu, Suu, Kuu, lf, lx, eps, kappa)); % This is the upper confidence bound acquisition function.
					zlabel('UCB acquisition function');
				elseif iAF == 3
					[~, fOpt] = optimizeAcquisitionFunction(@(x)(AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)), xMin, xMax, nStarts);
					AF = @(x)(AFPI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xi)); % This is the probability of improvement acquisition function.
					zlabel('PI acquisition function');
				else
					[~, fOpt] = optimizeAcquisitionFunction(@(x)(AFEV(x, Xu, muu, Suu, Kuu, lf, lx, eps)), xMin, xMax, nStarts);
					AF = @(x)(AFEI(x, Xu, muu, Suu, Kuu, lf, lx, eps, fOpt, xi)); % This is the expected improvement acquisition function.
					zlabel('EI acquisition function');
				end
				% We calculate the acquisition function values at the plot points and plot those.
				afValues = zeros(1,ns);
				for k = 1:ns
					afValues(k) = AF(Xs(:,k));
				end
				afValues(afValues < -1e100) = min(afValues(afValues > -1e100)); % We set the insanely small numbers (which may occur when the probability is pretty much zero) to the lowest not-insanely-small number.
				afValues = reshape(afValues, nsPerDimension, nsPerDimension); % We put the result in a square format again.
				meshc(x1Mesh,x2Mesh,afValues);
				surface(x1Mesh,x2Mesh,afValues);
				view([20,25]);
				if useColor == 0
					colormap(((0:0.01:0.8)'*ones(1,3)).^.8)
				else
					colormap('default');
				end
			end
			if exportFigs ~= 0
				if iAF == 1
				elseif iAF == 2
					export_fig(['UCBAcquisitionFunction2D',num2str(run),'.png'],'-transparent');
				elseif iAF == 3
					export_fig(['PIAcquisitionFunction2D',num2str(run),'.png'],'-transparent');
				else
					export_fig(['EIAcquisitionFunction2D',num2str(run),'.png'],'-transparent');
				end
			end
		end % End of check whether we should make plots.
	end % End of iterating over acquisition functions.
end % End of experiment runs.
disp(['We are done with all the experiments! The time passed is ',num2str(toc),'.']);

% We save all the data we have generated.
save('TwoDimensionalOptimizationData','sXm','sf','sfh','snu','sXu','smuu','sSuu','sKuu','sParticles','sWeights','sRecommendations','sRecommendationValue','sRecommendationBelievedValue');

%% We load the data for the plots. You can also run this block separately, although you would have to run the first block of the script too, to set up all the settings.
% load('TwoDimensionalOptimizationDataFiftyRuns');
load('TwoDimensionalOptimizationData');
nInputs = size(sXm, 2);
nAF = size(sXm, 3);
nRuns = size(sXm, 4);

% Now that we're done iterating, we calculate the instantaneous and cumulative regrets. Note that for the first we need to use the recommended points of the GPs (the highest mean) while for the
% latter we need to use the points that were actually tried out.
[xOptTrue, fOptTrue] = optimizeAcquisitionFunction(OptimizationFunction, xMin, xMax, nStarts); % Sometimes this optimization function gives the wrong optimum. When your graphs look odd, try running this block again.
meanRecommendationValues = mean(sRecommendationValue, 3);
meanError = fOptTrue - meanRecommendationValues;
meanObtainedValues = mean(sf, 3);
meanObtainedRegret = fOptTrue - meanObtainedValues;
meanRegret = cumsum(meanObtainedRegret, 1);

% We make a plot of the regret over time.
colors = [red;blue;yellow;green;grey];
figure(1);
clf(1);
hold on;
grid on;
for i = 1:nAF
	plot(0:nInputs, [0; meanRegret(:,i)], '-', 'Color', colors(i,:));
end
xlabel('Measurement number');
ylabel('Cumulative regret over time');
legend('TS','UCB','PI','EI','Location','SouthEast');
% axis([0,nInputs,0,50]);
if exportFigs ~= 0
	export_fig('CumulativeRegret2D.png','-transparent');
end

% We make a plot of the error over time.
figure(2);
clf(2);
hold on;
grid on;
for i = 1:nAF
	plot(0:nInputs, [fOptTrue - mean(fs); meanError(:,i)], '-', 'Color', colors(i,:));
end
xlabel('Measurement number');
ylabel('Recommendation error over time');
legend('TS','UCB','PI','EI','Location','NorthEast');
% axis([0,nInputs,0,0.5]);
if exportFigs ~= 0
	export_fig('RecommendationError2D.png','-transparent');
end

% Finally we display the error in the final recommendations.
disp('The average final recommendation errors were:');
disp(meanError(end,:));
