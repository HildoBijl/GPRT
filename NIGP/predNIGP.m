function [m S model] = predNIGP(model, xs, mode)
% [m S] = predNIGP(model, xs, mode)
% Function to make test predictions with the NIGP model.
% Inputs
%    model      struct returned by trainNIGP with fields
%      .x       training inputs, N-by-D
%      .y       training targets, N-by-E
%      .seard   squared exp kernel hyperparameters
%      .lsipn   log input noise standard deviation, D-by-1
%      .R       (optional) cholesky of training cov matrix, N-by-N-by E
%      .alpha   (optional) [K + sn2I + ipK]^-1 * y
%    xs         test inputs, Ns-by-D
%    mode       (0) don't use uncertainty, (1) use uncertainty, (2) use
%               analytic moment approach
%
% Outputs
%   m           predicted posterior means, Ns-by-E
%   S           predicted posterior variances, with noise, Ns-by-E
%   model       model with R and alpha fields
%
% For more information see the NIPS paper: Gaussian Process Training with
% Input Noise; McHutchon and Rasmussen, 2011. 
% http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon
%
% 17 Jan 2012, Andrew McHutchon

x = model.x; y = model.y; [N E] = size(y); if nargin < 3; mode = 0; end

if ~isfield(model,'R')
    XmX2 = bsxfun(@minus,permute(x,[1,3,2]),permute(x,[3,1,2])).^2;
    K = calcK(model.seard, XmX2);                        % N-by-N-by-E
    sn2I = bsxfun(@times,permute(exp(2*model.seard(end,:)),[1,3,2]),eye(N));% N-by-N-by-E
    ipK = dipK2ipK(model.dipK);
    Kn = K + sn2I + ipK;
    model.R = chol3(Kn);
end
if ~isfield(model,'alpha'); model.alpha = findAlpha(model.R,y); end

    
if 2 == mode
    [m S] = gpm(model,xs);
    S = bsxfun(@plus,S,exp(2*model.seard(end,:)));       % Add output noise
else
    [Kss, Kxs] = calcKtest(model, xs);                  %  test covariances
    
    m = etprod('12',Kxs,'312',model.alpha,'32');       % predicted means
    
    if nargout > 1                          % predictive variances as well
        ipKss = calcipKtest(model, xs, Kxs, mode);
        v = zeros(N,size(xs,1),E);
        for i=1:E; v(:,:,i) = model.R(:,:,i)'\Kxs(:,:,i); end  % N-by-Ns-by-E
        S = Kss - etprod('12',v,'312',v,'312');
        S = bsxfun(@plus,S,exp(2*model.seard(end,:)));    % add output noise
        S = S + ipKss;                                    % add input noise
    end
end
