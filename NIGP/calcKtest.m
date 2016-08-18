function [Kss Kxs] = calcKtest(model, xs)
% Function to calculate the two GP test point covariance matrices using the
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% lhyp = [log(ell_1); log(ell_2); ...; log(ell_D); log(sqrt(sf2))]
%
% Kss is the covariance of each test point with itself and Kxs is the
% covariance between each training point and each test point.
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)
% Modified, Andrew McHutchon, Jan 2012   

[Ns D] = size(xs); [N E] = size(model.y);                      % Dimensions
iell = exp(-model.seard(1:D,:));             % D-by-E, inverse lengthscales
sf2 = exp(2*model.seard(D+1,:));                  % 1-by-E, signal variance
x = model.x;

Kss = repmat(sf2,size(xs,1),1); % Ns-by-E, cov of test points with themselves
Kxs = zeros(N,Ns,E);
for i=1:E
    Kxs(:,:,i) = sf2(i)*exp(-sq_dist(diag(iell(:,i))*x',diag(iell(:,i))*xs')/2);
end