function dK = derivK(lh, K, XmX2)
% dK = derivK(lh, K, XmX2)
% Function to calculate the derivative of the training set covariance matrix
% w.r.t the D log lengthscales and the log signal std hyperparameter.
%   Inputs:
%           lh = D+2-by-1, K = N-by-N, XmX2 = N-by-N-by-D
%
% The derivative is given by,
%   dK = iell2 .* XmX2 .* K         N-by-N-by-D+1
%
% Andrew McHutchon, July 2012

N = size(K,1); D = length(lh)-2; 
if size(lh,2) > 1; error('derivK only for E = 1\n'); end

iell2 = permute(exp(-2*lh(1:D)),[3,2,1]); % 1-by-1-by-D

dK = zeros(N,N,D+1);
dK(:,:,1:D) = bsxfun(@times,iell2(1,1,:),XmX2);
dK(:,:,1:D) = bsxfun(@times,K,dK(:,:,1:D));

dK(:,:,D+1) = 2*K;