function Kss = calcipKtest(model, xs, Kxs, mode)
% Kss = calcipKtest(model, xs, Kxs, mode)
% Function to calculate the additional variance on a test prediction
% stemming from the expected noise on the test inputs.
% Inputs
%   model       struct containing the fields
%     .x        training inputs
%     .alpha    [Kxx + sn2I + ipK]^-1 * y
%   xs          test inputs, Ns-by-D
%   Kxs         covariance matrix between training and test inputs
%   mode        whether to include the uncertainty (1) or not (0)
%
% Andrew McHutchon, 17/01/2012

x = model.x; [N D] = size(x); alpha = model.alpha;

s2(1,1,1:D) = exp(2*model.lsipn);            % 1-by-1-by-D, noise variances
iell2 = exp(-2*model.seard(1:D,:));   % D-by-E, inverse squared lengthscales

Xmx = permute(bsxfun(@minus,permute(x,[1,3,2]),permute(xs,[3,1,2])),[1,2,4,3]); % N-by-Ns-by-1-by-D
XmxiLam = bsxfun(@times,Xmx,permute(iell2,[4,3,2,1]));  % N-by-Ns-by-E-by-D
dKxsdx = bsxfun(@times,XmxiLam,Kxs);      % N-by-Ns-by-E-by-D, deriv of Kxs
   
if 0 == mode                 % not including uncertainty in the derivatives
    dfs2 = etprod('123',dKxsdx,'4123',alpha,'42').^2;       % Ns-by-E-by-D       

else                             % including uncertainty in the derivatives    
    aa = etprod('123',alpha,'13',alpha,'23') - solve_cholE(model.R,eye(N)); % N-by-N-by-E
    ilsf = exp(2*bsxfun(@minus,model.seard(D+1,:),model.seard(1:D,:))); % D-by-E

    dfs2 = etprod('123',etprod('1234',dKxsdx,'5134',aa,'523'),'1423',dKxsdx,'4123'); % Ns-by-E-by-D
    dfs2 = bsxfun(@plus,permute(ilsf,[3,2,1]),dfs2);
end

Kss = sum(bsxfun(@times,dfs2,s2),3); % Ns-by-E, squared slope x noise variance
