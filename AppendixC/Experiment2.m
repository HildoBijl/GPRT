% This file contains the experiment 2 of Appendix C of the Gaussian process regression thesis. We take a simple LQG system and find a proper controller (feedback matrix).
% To use it, make sure that the Matlab directory is set to the directory of this file. Then you can run this file. At the end data is saved. It is also possible to load data there and only
% execute the last bit of script for yourself.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 1; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

% We fix Matlab's random number generator, so that it always gives the same result as the figure in the thesis.
rng(1, 'twister');

% We define the system.
A = [1,0;5e-2,1];
B = [1;0];
V = [1,0;0,1];

% We define the initial conditions.
mu0 = [1;0]; % We initialize the system in a controllable state.
Sig0 = [1,0;0,0]*1e-1;
Psi0 = Sig0 + mu0*mu0';

% We set up the cost function including its discount exponent.
a = -0.8;
Q = eye(2);
R = eye(1);

% We set up some auxiliary matrices.
n = size(A,1);
I = eye(n);
Z = zeros(n,n);
Aa = A + a*I;

% We calculate the optimal controller and set up the controlled matrix.
X = are(Aa, B/R*B', Q);
Fopt = R\B'*X;
Ac = A - B*Fopt;
Qc = Q + Fopt'*R*Fopt;

% We calculate the cost mean and variance.
XbaQ = lyap((Ac+a*I)',Qc);
X2aPsi0 = lyap((Ac+2*a*I),Psi0);
X2aV = lyap((Ac+2*a*I),V);
EJfin = trace((Psi0 - V/(2*a))*XbaQ);
VJfin = 2*trace((Psi0*XbaQ)^2) - 2*(mu0'*XbaQ*mu0)^2 + 4*trace((X2aPsi0 - X2aV/(4*a))*XbaQ*V*XbaQ);

% We display the mean and the std.
disp(['The optimal cost has mean ',num2str(EJfin),' and STD ',num2str(sqrt(VJfin)),'.']);
disp(['It occurs at F_1=',num2str(Fopt(1)),', F_2=',num2str(Fopt(2)),'.']);

% We now set up a new controller minimizing the variance.
options = optimset('Display', 'off') ; % We do not want any output from fmincon.
Fmv = fmincon(@(F)(getCostVariance(A,B,F,V,mu0,Psi0,a,Q,R)), Fopt, [], [], [], [], [], [], [], options);
Ac = A - B*Fmv;
Qc = Q + Fmv'*R*Fmv;

% We calculate the cost mean and variance.
XbaQ = lyap((Ac+a*I)',Qc);
X2aPsi0 = lyap((Ac+2*a*I),Psi0);
X2aV = lyap((Ac+2*a*I),V);
EJmv = trace((Psi0 - V/(2*a))*XbaQ);
VJmv = 2*trace((Psi0*XbaQ)^2) - 2*(mu0'*XbaQ*mu0)^2 + 4*trace((X2aPsi0 - X2aV/(4*a))*XbaQ*V*XbaQ);

% We display the mean and the std.
disp(['The minimum variance cost has mean ',num2str(EJmv),' and STD ',num2str(sqrt(VJmv)),'.']);
disp(['It occurs at F_1=',num2str(Fmv(1)),', F_2=',num2str(Fmv(2)),'.']);

% We now start the process of making data for graph. First we define some settings.
numPlotPoints = 101;
Fopt = R\B'*X;
fMin = -0.4;
fMax = 1.6;
fSpread = fMin:((fMax-fMin)/(numPlotPoints-1)):fMax;

% We define storage matrices for EJ and VJ and then fill them up.
EJPlot = zeros(size(fSpread));
VJPlot = zeros(size(fSpread));
for i = 1:numPlotPoints
	% We set up the controller for the new system.
	F = (1 - fSpread(i))*Fopt + fSpread(i)*Fmv;
	Ac = A - B*F;
	Qc = Q + F'*R*F;

	% We check if the resulting cost will be finite.
	if (max(real(eig(Ac+a*I))) < 0)
		% We calculate the cost mean and variance.
		XbaQ = lyap((Ac+a*I)',Qc);
		X2aPsi0 = lyap((Ac+2*a*I),Psi0);
		X2aV = lyap((Ac+2*a*I),V);
		EJPlot(i) = trace((Psi0 - V/(2*a))*XbaQ);
		VJPlot(i) = 2*trace((Psi0*XbaQ)^2) - 2*(mu0'*XbaQ*mu0)^2 + 4*trace((X2aPsi0 - X2aV/(4*a))*XbaQ*V*XbaQ);
	else
		% We set the cost mean and variance to -1 to indicate infinity.
		EJPlot(i) = -1;
		VJPlot(i) = -1;
	end
end
% When there were invalid controllers, we set the corresponding cost to the maximum we found of the valid ones. This keeps the plots at least somewhat reasonable.
for i = 1:numPlotPoints
	EJMax = max(max(EJPlot));
	if EJPlot(i) == -1
		EJPlot(i) = EJMax;
	end
	VJMax = max(max(VJPlot));
	if VJPlot(i) == -1
		VJPlot(i) = VJMax;
	end
end
stdJPlot = sqrt(VJPlot);

% We plot the mean and the std in one plot.
figure(1);
clf(1);
hold on;
grid on;
if useColor == 0
	plot(fSpread,EJPlot,'k-');
	plot(fSpread,stdJPlot,'k--');
else
	plot(fSpread,EJPlot,'b-');
	plot(fSpread,stdJPlot,'r-');
end
xlabel('Controller factor f');
ylabel('Mean and standard deviation of the cost J');
legend('Mean','Standard deviation');
axis([fMin,fMax,150,230]);

% We export the plot, if desired.
if exportFigs ~= 0
	export_fig('CostMeanAndStdForVaryingController.png','-transparent');
end

% Next, we will run simulations. The goal is to come up with a histogram of the cost, and see how many of the points are above a certain threshold.
% We set which controller we will use.
F = [Fopt;Fmv];
m = size(F,1);
for i = 1:m
	Ac(:,:,i) = A - B*F(i,:);
	Qc(:,:,i) = Q + F(i,:)'*R*F(i,:);
end

% We first examine what time steps we need to take.
T = 20;
dt = 0.01;
t = 0:dt:T;
numTimeSteps = length(t)-1;
numExperiments = 2000;

% We calculate what the expected cost is.
EJfin = zeros(1,m);
VJfin = zeros(1,m);
EJinf = zeros(1,m);
VJinf = zeros(1,m);
for i = 1:m
	% We calculate Lyapunov matrices and other important matrices to calculate the cost.
	XV = lyap(Ac(:,:,i),V); % X^V
	Delta = Psi0 - XV; % \Delta
	PsiT = expm(Ac(:,:,i)*T)*(Psi0 - XV)*expm(Ac(:,:,i)'*T) + XV; % \Psi(T)
	Aa = Ac(:,:,i) + a*I; % A_\alpha
	A2a = Ac(:,:,i) + 2*a*I; % A_{2\alpha}
	Ama = Ac(:,:,i) - a*I; % A_{-\alpha}
	XbaQ = lyap(Aa',Qc(:,:,i)); % \bar{X}_\alpha^Q
	XbaQT = XbaQ - expm(Aa'*T)*XbaQ*expm(Aa*T); % \bar{X}_\alpha^Q(T)
	XbmaQ = lyap(Ama',Qc(:,:,i)); % \bar{X}_{-\alpha}^Q
	XbmaQT = XbmaQ - expm(Ama'*T)*XbmaQ*expm(Ama*T); % \bar{X}_{-\alpha}^Q(T)
	X2aD = lyap(A2a,Delta); % X_{2\alpha}^\Delta
	X2atT = [I,Z]*expm([A2a,X2aD*expm(A2a'*T)*Qc(:,:,i);Z,Ac(:,:,i)]*T)*[Z;I];
	X2aPsi0 = lyap((Ac(:,:,i)+2*a*I),Psi0);
	X2aV = lyap((Ac(:,:,i)+2*a*I),V);

	% We calculate the mean and variance for the current alpha, both in the infinite-time and the finite-time case.
	EJinf(i) = trace((Psi0 - V/(2*a))*XbaQ);
	VJinf(i) = 2*trace((Psi0*XbaQ)^2) - 2*(mu0'*XbaQ*mu0)^2 + 4*trace((X2aPsi0 - X2aV/(4*a))*XbaQ*V*XbaQ);
	EJfin(i) = trace((Psi0 - exp(2*a*T)*PsiT + (1 - exp(2*a*T))*(-V/(2*a)))*XbaQ);
	VJfin(i) = 2*trace((Delta*XbaQT)^2) - 2*(mu0'*XbaQT*mu0)^2 + 4*trace(XV*Qc(:,:,i)*(XV*(exp(4*a*T)*XbmaQT - XbaQT)/(4*a) + 2*X2aD*XbaQT - 2*X2atT));
end

% We calculate matrices for a system discretization.
for i = 1:m
	Ad(:,:,i) = expm(Ac(:,:,i)*dt); % This is the discrete-time A matrix.
	XV(:,:,i) = lyap(Ac(:,:,i),V);
	XVdt(:,:,i) = XV(:,:,i) - expm(Ac(:,:,i)*dt)*XV(:,:,i)*expm(Ac(:,:,i)'*dt);
	Vd(:,:,i) = XVdt(:,:,i); % This is the discrete-time noise matrix.
	VdChol(:,:,i) = chol(Vd(:,:,i)); % The cholesky decomposition.
end

% We prepare storage matrices.
state = zeros(n,m,numTimeSteps+1);
cost = zeros(numExperiments,m);

% The time has come to do the experiments.
tic;
disp('Running simulations...');
for i = 1:numExperiments
	if mod(i,numExperiments/100) == 0
		disp(['We are on ',num2str(i/numExperiments*100),'%. Time passed is ',num2str(toc),' seconds. Estimated time left is ',num2str(toc/i*(numExperiments-i)),' seconds.']);
	end
	% We set up an initial state. It is the same for all experiments.
	initialState = mvnrnd(mu0,Sig0)';
	for j = 1:m
		state(:,j,1) = initialState;
		cost(i,j) = 0;
	end
	
	% We take the required number of timesteps, each time updating the
	% state and adding to the cost for every value of alpha.
	for j = 1:numTimeSteps
		noiseAddition = randn(n,1);
		for k = 1:m
			state(:,k,j+1) = Ad(:,:,k)*state(:,k,j) + VdChol(:,:,k)*noiseAddition;
			cost(i,k) = cost(i,k) + (exp(2*a*t(j))*state(:,k,j)'*Qc(:,:,k)*state(:,k,j) + exp(2*a*t(j+1))*state(:,k,j+1)'*Qc(:,:,k)*state(:,k,j+1))*dt/2;
		end
	end
end

% We save the results.
save('Experiment2Data', 'cost', 'numExperiments', 'F', 'EJfin', 'VJfin', 'EJinf', 'VJinf');

% We display the results about the cost.
threshold = 1500;
costMean = mean(cost);
costStd = std(cost);
disp(['- Calculations are finished for ',num2str(numExperiments),' experiments - ']);
for i = 1:size(F,1)
	disp(['Examining the case where F = ',num2str(F(i,:)),'.']);
	disp(['The cost had mean ',num2str(costMean(i)),' with an std of ',num2str(costStd(i)),'.']);
	disp(['The finite-time predicted mean cost was ',num2str(EJfin(i)),' with an std of ',num2str(sqrt(VJfin(i))),', giving a mean+2*std of ',num2str(EJfin(i)+2*sqrt(VJfin(i))),'.']);
	disp(['The infinite-time predicted mean cost was ',num2str(EJinf(i)),' with an std of ',num2str(sqrt(VJinf(i))),', giving a mean+2*std of ',num2str(EJinf(i)+2*sqrt(VJinf(i))),'.']);
	disp(['A percentage of ',num2str(sum(cost(:,i)>threshold)/numExperiments*100),'% of the costs is larger than ',num2str(threshold),'.']);
end