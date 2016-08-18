function df2 = calcdf2(lhyp,x,preC)
% df = calcdf(lhyp,x,preC)
% Function to calculate the derivative of the posterior mean of a GP about
% each of the training points.
% Inputs
%   lhyp          the log hyperparameters, struct with two fields
%       .seard    hyperparameters for the squared exponential kernal,
%                  D+2-by-E
%   x             training inputs matrix, N-by-D
%   preC          pre computed variables
%       .XmX      distance between all pairs of data points, N-by-N-by-D
%       .K        training set covariance matrix (without noise), N-by-N-by-E
%       .alpha    Kn^-1 * y, N-by-E
%
% Output
%   df2            matrix of squared derivatives, N-by-D
%
% Andrew McHutchon, July 2012

[N D] = size(x); E = size(lhyp.seard,2);

iell2 = exp(-2*lhyp.seard(1:D,:));                 % squared lengthscales D-by-E
df2 = zeros(N,E,D);

% Form the derivative covariance function
for i=1:E                                           % loop over E to save memory
    XmXiLam = bsxfun(@times,preC.XmX,permute(iell2(:,i),[3,2,1]));% N-by-N-by-D
    dKdx = bsxfun(@times,XmXiLam,preC.K(:,:,i));                  % N-by-N-by-D

    % Compute derivative
    df2(:,i,:) = etprod('12',dKdx,'412',preC.alpha(:,i),'4').^2;  % N-by-D
end
