% This file contains the experiment 1 of Appendix C of the Gaussian process regression thesis. We take a simple LQG system and run it a ton of times to see what the resulting cost function is.
% To use it, make sure that the Matlab directory is set to the directory of this file. Then you can run this file. The first half of this file runs the experiments and creates a data file. The
% second half of this file then processes and plots the data.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.

% We fix Matlab's random number generator, so that it always gives the same result as the figure in the thesis.
rng(1, 'twister');

% We first define the system and the cost function set-up.
A = [1/4,1/2;-1/7,-1/3];
mu0 = [1;-1];
Sig0 = [4,1;1,2];
Psi0 = Sig0 + mu0*mu0';
V = [3,-2;-2,8];
Q = eye(2);
T = 3; % This is the simulation time. You can increase this by a factor 100 (and increase dt as well to prevent too much waiting time) to see the matrix exponential method fail.
numExperiments = 1000000;
dt = 0.01;

% We set up some auxiliary parameters.
n = size(A,1);
I = eye(n);
Z = zeros(n,n);

% We calculate some important system properties that do not depend on alpha.
XV = lyap(A,V); % X^V
Delta = Psi0 - XV; % \Delta
PsiT = expm(A*T)*(Psi0 - XV)*expm(A'*T) + XV; % \Psi(T)

% We calculate the system mean and expected squared value after T seconds for various alpha values.
alpha = [0.05,0,-0.05,-0.1];
numAlpha = length(alpha);
costMeanPrediction = zeros(size(alpha));
costVariancePrediction = zeros(size(alpha));
costMeanPrediction2 = zeros(size(alpha));
costVariancePrediction2 = zeros(size(alpha));
for i = 1:numAlpha
	% We extract the value for alpha.
	a = alpha(i);
	% We calculate the properties through Lyapunov solutions.
	if a == 0
		% We calculate Lyapunov matrices for the zero alpha.
		XbQ = lyap(A',Q); % \bar{X}^Q
		XbQT = XbQ - expm(A'*T)*XbQ*expm(A*T); % \bar{X}^Q(T)
		XbXbQ = lyap(A',XbQ); % \bar{X}^{\bar{X}^Q}
		XbXbQT = XbXbQ - expm(A'*T)*XbXbQ*expm(A*T); % \bar{X}^{\bar{X}^Q}(T)
		XD = lyap(A,Delta); % X^\Delta
		XtT = [I,Z]*expm([A,XD*expm(A'*T)*Q;Z,A]*T)*[Z;I];
		
		% We calculate the mean and variance for the current alpha.
		costMeanPrediction(i) = trace((Psi0 - PsiT + T*V)*XbQ);
		costVariancePrediction(i) = 2*trace((Delta*XbQT)^2) - 2*(mu0'*XbQT*mu0)^2 + 4*trace(XV*Q*(XV*(T*XbQ - XbXbQT) + 2*XD*XbQT - 2*XtT));
	else
		% We calculate Lyapunov matrices for the current alpha.
		Aa = A + a*I; % A_\alpha
		A2a = A + 2*a*I; % A_{2\alpha}
		Ama = A - a*I; % A_{-\alpha}
		XbaQ = lyap(Aa',Q); % \bar{X}_\alpha^Q
		XbaQT = XbaQ - expm(Aa'*T)*XbaQ*expm(Aa*T); % \bar{X}_\alpha^Q(T)
		XbmaQ = lyap(Ama',Q); % \bar{X}_{-\alpha}^Q
		XbmaQT = XbmaQ - expm(Ama'*T)*XbmaQ*expm(Ama*T); % \bar{X}_{-\alpha}^Q(T)
		X2aD = lyap(A2a,Delta); % X_{2\alpha}^\Delta
		X2atT = [I,Z]*expm([A2a,X2aD*expm(A2a'*T)*Q;Z,A]*T)*[Z;I];
		
		% We calculate the mean and variance for the current alpha.
		costMeanPrediction(i) = trace((Psi0 - exp(2*a*T)*PsiT + (1 - exp(2*a*T))*(-V/(2*a)))*XbaQ);
		costVariancePrediction(i) = 2*trace((Delta*XbaQT)^2) - 2*(mu0'*XbaQT*mu0)^2 + 4*trace(XV*Q*(XV*(exp(4*a*T)*XbmaQT - XbaQT)/(4*a) + 2*X2aD*XbaQT - 2*X2atT));
	end
	% We calculate the properties through matrix exponentials.
	Aa = A + a*I; % A_\alpha
	A2a = A + 2*a*I; % A_{2\alpha}
	Am2a = A - 2*a*I; % A_{-2\alpha}
	C = [-A2a',Q,Z,Z,Z;Z,A,V,Z,Z;Z,Z,-A',Q,Z;Z,Z,Z,A2a,V;Z,Z,Z,Z,-Am2a'];
	Ce = expm(C*T);
	C44 = Ce(7:8,7:8);
	C12 = Ce(1:2,3:4);
	C13 = Ce(1:2,5:6);
	C14 = Ce(1:2,7:8);
	C15 = Ce(1:2,9:10);
	costMeanPrediction2(i) = trace(C44'*(C12*Psi0 + C13));
	costVariancePrediction2(i) = 2*trace((C44'*(C12*Psi0 + C13))^2 - 2*C44'*(C14*Psi0 + C15)) - 2*(mu0'*C44'*C12*mu0)^2;
end
costStdPrediction = sqrt(costVariancePrediction);
costStdPrediction2 = sqrt(costVariancePrediction2);

% The next step is to do experiments with the system. We first examine what time steps we need to take.
t = 0:dt:T;
numTimeSteps = length(t)-1;

% We calculate matrices for a system discretization.
Ad = expm(A*dt); % This is the discrete-time A matrix.
XVdt = XV - expm(A*dt)*XV*expm(A'*dt);
Vd = XVdt; % This is the discrete-time noise matrix.
VdChol = chol(Vd); % This is the cholesky decomposition of the discrete-time noise matrix.

% We prepare storage matrices.
state = zeros(n,numTimeSteps+1,numExperiments);
cost = zeros(numExperiments,numAlpha);

% The time has come to do the experiments.
tic;
for i = 1:numExperiments
	if mod(i,numExperiments/100) == 0
		disp(['We are on ',num2str(i/numExperiments*100),'%. Time passed is ',num2str(toc),' seconds. Estimated time left is ',num2str(toc/i*(numExperiments-i)),' seconds.']);
	end
	% We set up an initial state.
	state(:,1,i) = mvnrnd(mu0,Sig0)';
	cost(i,:) = 0;
	
	% We take the required number of timesteps, each iteration updating the state and adding to the cost for every value of alpha.
	for j = 1:numTimeSteps
		state(:,j+1,i) = Ad*state(:,j,i) + VdChol*randn(n,1);
		cost(i,:) = cost(i,:) + (exp(2*alpha*t(j))*(state(:,j,i)'*Q*state(:,j,i)) + exp(2*alpha*t(j+1))*(state(:,j+1,i)'*Q*state(:,j+1,i)))*dt/2;
	end
end

% We display the results about the cost.
costMean = mean(cost);
costStd = std(cost);
disp(['- Calculations are finished for ',num2str(numExperiments),' experiments - ']);
for k = 1:numAlpha
	a = alpha(k);
	disp(['Displaying results for alpha = ',num2str(a),'.']);
	disp(['The cost had mean ',num2str(costMean(k)),' with an std of ',num2str(costStd(k)),'.']);
% 	disp(['The predicted mean cost was ',num2str(costMeanPrediction(k)),' with an std of ',num2str(costStdPrediction(k)),' according to Lyapunov solutions.']);
% 	disp(['The predicted mean cost was ',num2str(costMeanPrediction2(k)),' with an std of ',num2str(costStdPrediction2(k)),' according to matrix exponentials.']);
end

% We save the data which we just generated.
save('Experiment1Data','cost','alpha');

%% Now it is time to plot the PDFs based on the data that we gathered.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 1; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

% We load the experiment data and calculate some parameters.
load('Experiment1Data');
numAlpha = length(alpha);
numExperiments = size(cost,1);
costMean = mean(cost);
costStd = std(cost);

% We define the bins. There are two different bin sizes, one for low costs and one for high costs. This allows us to get more resolution for the lower costs where the peaks are.
numLowBins = 15; % This is the number of bins in the low cost region.
numHighBins = 24; % This is the number of bins in the high cost region.
binSwitch = 40; % This is the turnaround point for the bin sizes.
upperLimit = 150; % This is the highest bin position.
lowBinSize = binSwitch/numLowBins;
highBinSize = (upperLimit - binSwitch)/(numLowBins - 1/2);

% We set up the bins. From this histogram we derive the probability density function.
binEdges = [-0.01,0.01,linspace(lowBinSize,binSwitch,numLowBins),linspace(binSwitch + highBinSize, upperLimit + highBinSize/2, numHighBins)];
numPerBin = histc(cost, binEdges)';
binWidths = binEdges(2:end) - binEdges(1:end-1);
probability = numPerBin(:,1:end-1)/numExperiments./repmat(binWidths,numAlpha,1); % This is the probability density function of the cost.
binCenters = (binEdges(1:end-1) + binEdges(2:end))/2;

% We start setting up the plot of the cost PDF.
figure(1);
clf(1);
hold on;
grid on;
if useColor == 0
	plotMode = {'k-','k--','k:','k-.'};
else
	plotMode = {'r-','k-','b-','g-'};
end
legendCollector = {};

% We plot the probability distribution of the cost function.
for i = 1:numAlpha
	plot(binCenters, probability(i,:), plotMode{i}, 'LineWidth', 1);
	legendCollector{i} = ['\alpha=',num2str(alpha(i))];
end

% We plot some auxiliary data, like the means of the PDFs.
for i = 1:numAlpha
	% We extract the mean and standard deviation, and look at which
	% probability there is at the specific point.
	currMean = costMean(i);
	currStd = costStd(i);
	currMeanProb = interp1(binCenters,probability(i,:),currMean);
	% We plot the mean of the cost PDF.
	plot([currMean,currMean], [0,currMeanProb], plotMode{i});
end

% We set up the axes and legend.
xlabel('Cost J_T');
ylabel('Probability density');
legend(legendCollector);
axis([0,upperLimit,0,0.018]);

% We export the plot, if desired.
if exportFigs ~= 0
	export_fig('CostPDF.png','-transparent');
end