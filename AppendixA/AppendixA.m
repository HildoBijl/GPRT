% This file contains all the scripts for Appendix A of the Gaussian process regression thesis. 
% To use it, make sure that the Matlab directory is set to the directory of this file. Then you can run this file. It is not very extensive, because there is only one experiment in this appendix.

% We set up the workspace, ready for executing scripts.
clear all; % Empty the workspace.
clc; % Empty the command window.
exportFigs = 1; % Do we export figures? 0 for no, 1 (or anything else) for yes.
useColor = 1; % Should we set up plots for colored output (1) or black-and-white output (0)?
addpath('../ExportFig'); % We add the functions for exporting figures.

% We fix Matlab's random number generator, so that it always gives the same result as the figure in the thesis.
rng(5, 'twister');

% We start off by defining some settings.
nExp = 100; % How many experiments do we run?
n = 3; % This is the size of the matrices used.
timeExponents = -2.5:0.1:1.5; % Which exponents for time shall we use? We will subsequently use the times 10^timeExponents.

% We set up storage matrices.
comparison = zeros(1,length(timeExponents));
comparisonStd = zeros(1,length(timeExponents));
results = zeros(2,nExp);

% We loop through the time parameters.
for i = 1:length(timeExponents)
	% This is the time parameter which we use.
	t = 10^timeExponents(i); 
	
	% We start doing experiments.
	j = 1;
	while j <= nExp
		% We define random matrices for our Lyapunov equation.
		A = randn(n,n);
		Q = randn(n,n);

		% Now we calculate the result by directly solving the Lyapunov equation.
		XQ = lyap(A,Q);
		XQtLyap = XQ - expm(A*t)*XQ*expm(A'*t);

		% And we calculate the result through a matrix exponential.
		XQtExp = expm(A*t)*[eye(n),zeros(n)]*expm([-A,Q;zeros(n),A']*t)*[zeros(n);eye(n)];

		% Now we find some measure of the numerical error.
		resultLyap = sum(sum(abs(A*XQtLyap + XQtLyap*A' + Q - expm(A*t)*Q*expm(A'*t))));
		resultExp = sum(sum(abs(A*XQtExp + XQtExp*A' + Q - expm(A*t)*Q*expm(A'*t))));

		% And we store the results.
		results(:,j) = [resultLyap;resultExp];
		
		% Finally we check if any of the results is exactly 0. If so, we redo this experiment, because if messes up our computation. And a zero error does not happen in reality either. If not,
		% we continue with the next experiment.
		if resultLyap ~= 0 && resultExp ~= 0
			j = j + 1;
		end
	end

	% We calculate the ratio of the results and then find the mean between them.
	ratios = results(2,:)./results(1,:);
	logRatios = log10(ratios);
	comparison(i) = mean(logRatios);
	comparisonStd(i) = std(logRatios)/sqrt(nExp);
	
	if comparison(i) > 1e4
		break;
	end
end

% We set up an errorbar plot based on the results.
figure(1);
clf(1);
hold on;
grid on;
if useColor == 0
	errorbar(timeExponents, comparison, 2*comparisonStd, 2*comparisonStd, 'k-');
else
	errorbar(timeExponents, comparison, 2*comparisonStd, 2*comparisonStd, 'b-');
end
xlabel('log_{10}(t)');
ylabel('log_{10}(m.e. error/L.s. error)');
axis([timeExponents(1)-.5,timeExponents(end)+.5,-2,6]);

% We export the plot, if desired.
if exportFigs ~= 0
	export_fig('AccuracyTest.png','-transparent');
end