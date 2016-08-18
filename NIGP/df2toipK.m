function ipK = df2toipK(lh,df2)
% Function to convert input noise variances and slopes into a N-by-N-by-E
% covariance matrix, writen diag{ Delta * Sigma_x * Delta^T } in the paper,
% and called ipK in the code. Note df2 is not recalculated.
% Inputs
%   lh          struct of hyperparameters
%     .lsipn    D-by-1, log standard deviations of input noise
%
% Andrew McHutchon, 16/01/2012

if nargin == 1; df2 = lh.df2; end
[N E D] = size(df2);
s2(1,1,1:D) = exp(2*lh.lsipn); % 1-by-1-by-D, noise variances

ipK = zeros(N,N,E); didx = zeros(1,N*E);
for i=1:E; didx(1+(i-1)*N:N+(i-1)*N) = 1+(i-1)*N^2:N+1:N^2*i; end

ipK(didx) = sum(bsxfun(@times,df2,s2),3);         % N-by-N-by-E