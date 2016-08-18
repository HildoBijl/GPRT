function lp = logNormP(x,mu,Sigma)
% Function to calculate the log normal probability of a set of 1D points 
% with a given set of means and variances
% Inputs
%   x       N-by-M matrix of points
%   mu      N-by-M matrix of means or 1-by-M to use same mean for
%           each point in a column
%   Sigma   N-by-M matrix of variances or 1-by-M as above
%
% Output
%   lp      N-by-M log probabilities
% 
% Andrew McHutchon

xmmu2 = bsxfun(@minus,x,mu).^2; % N-by-M

lp = -0.5*(log(2*pi) + log(Sigma) + bsxfun(@rdivide,xmmu2,Sigma));

end

