function K = calcK(lh,XmX2)
% K = calcK(lh, XmX2)
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
% For speed it uses a pre-computed matrix XmX2 which are the squared differences
% between all points in the training set, N-by-N-by-D
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)
% Modified, Andrew McHutchon, July 2012

N = size(XmX2,1); [D E] = size(lh); D = D-2;
iell2 = exp(-2*lh(1:D,:));                        % D-by-E, inverse length scale
lsf2 = 2*lh(D+1,:);                               % 1-by-E, signal variance

XmX2iell2 = reshape(XmX2,[N^2,D])*iell2; % N^2-by-E

% Calculate covariance matrix
K = reshape(exp(bsxfun(@minus,lsf2,0.5*XmX2iell2)),[N,N,E]);