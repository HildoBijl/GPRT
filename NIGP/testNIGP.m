function [ae ll ipn] = testNIGP(x,y,xs,ys)
% Function to quickly and easily train NIGP and test it on a dataset. Both
% training modes and both prediction modes are tested and compared. If the
% function is called with two input arguments the dataset is split in half
% for training and testing after first randomising the order of the points.
% If four input arguments are given then the first two define the training
% set and the second two the test set. If no inputs are provided a data set
% is created. The function returns and prints out the absolute error, the 
% log marginal likelihood, and the learnt input input noise level for 
% predictions on the test set.
%
% Andrew McHutchon, Dec 2011

try addpath tprod; addpath util; catch; end

% If two input arguments, split into training and test set
if nargin < 4
    if 0 == nargin
        x = randn(400,2); 
        y = sin(x(:,1)) + cos(x(:,2)) + 0.1*randn(400,1);
        x = x + 0.2*randn(400,2);
    end
    N = size(x,1); Ntr = round(N/2);
    ridx = randperm(N); x = x(ridx,:); y = y(ridx,:);
    xs = x(Ntr+1:end,:); ys = y(Ntr+1:end,:);
    x = x(1:Ntr,:); y = y(1:Ntr,:);
end

% Train the hyperparameters by maximising marginal likelihood
hypNIGP = trainNIGP(x,y,-500,0);
hypNIGPu = trainNIGP(x,y,-500,1);

% Make predicitions
[ynu, s2nu] = predNIGP(hypNIGPu, xs, 0);
[ynum, s2num] = predNIGP(hypNIGPu, xs, 2);
[yn, s2n] = predNIGP(hypNIGP, xs, 1);
[ynm, s2nm] = predNIGP(hypNIGP, xs, 2);
fprintf('\n     Test Results\n\n')

% Absoulte test error
ae(1) = sum(mean(abs(ys-yn),1)); ae(2) = sum(mean(abs(ys-ynm),1));
ae(3) = sum(mean(abs(ys-ynu),1)); ae(4) = sum(mean(abs(ys-ynum),1));
fprintf('Absolute Error\n'); 
fprintf('NIGPu:\t\t %f\n',ae(3)); fprintf('NIGPu gpm:\t %f\n',ae(4));
fprintf('NIGP:\t\t %f\n',ae(1)); fprintf('NIGP gpm:\t %f\n\n',ae(2));

% Test log likelihood
ll(1) = sum(sum(logNormP(ys,yn,s2n))); ll(2) = sum(sum(logNormP(ys,ynm,s2nm)));
ll(3) = sum(sum(logNormP(ys,ynu,s2nu))); ll(4) = sum(sum(logNormP(ys,ynum,s2num)));
fprintf('Log test set marginal likelihood\n');
fprintf('NIGPu:\t\t %f\n',ll(3)); fprintf('NIGPu gpm:\t %f\n',ll(4));
fprintf('NIGP:\t\t %f\n',ll(1)); fprintf('NIGP gpm:\t %f\n\n',ll(2));

% Input noise standard deviation
ipn(1,:) = exp(hypNIGP.lsipn); ipn(2,:) = exp(hypNIGPu.lsipn);
fprintf('Learnt noise standard deviation \n');
fprintf('NIGPu:\t'); fprintf('%6.4f ',ipn(2,:));
fprintf('\nNIGP:\t'); fprintf('%6.4f ',ipn(1,:)); fprintf('\n');
