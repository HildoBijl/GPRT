function df2 = calcdf2u(lhyp,x,preC,xs)
% df = calcdfu(lhyp,x,preC,xs)
% Function to calculate the derivative of the posterior of a GP, including 
% the uncertainty, about each of the training points or a set of test points
% Inputs
%   lhyp          the log hyperparameters, struct with two fields
%       .seard    hyperparameters for the squared exponential kernal,
%                  D+2-by-E
%       .lsipn    log standard deviation input noise hyperparameters, one
%                 per input dimension
%   x             training inputs matrix, N-by-D
%   preC          pre computed variables
%   xs            a set of test points at which to evaluate the derivative
%
% Output
%   df            matrix of derivatives, N-by-D
%
% Andrew McHutchon, Dec 2011

[N D] = size(x); E = size(lhyp.seard,2);

iell2 = exp(-2*lhyp.seard(1:D,:));      % D-by-E

A = etprod('123',preC.alpha,'13',preC.alpha,'23') - solve_cholE(preC.R,eye(N)); % N-by-N-by-E
ilsf = exp(2*bsxfun(@minus,lhyp.seard(D+1,:),lhyp.seard(1:D,:))); % D-by-E

if nargin < 4
    df2 = zeros(N,E,D);
    for i=1:E
        % Training set
        % Find the derivative of the covariance function
        XmXiLam = bsxfun(@times,preC.XmX,permute(iell2(:,i),[3,2,1]));    % N-by-Ns-by-D
        dKdx = bsxfun(@times,XmXiLam,preC.K(:,:,i));                        % N-by-Ns-by-D

        % Compute derivative
        df2i = etprod('12',etprod('123',dKdx,'513',A(:,:,i),'52'),'142',dKdx,'412');       % Ns-by-D
        df2(:,i,:) = bsxfun(@plus,ilsf(:,i)',df2i);
    end

else
    % Test set
    Ns = size(xs,1); df2 = zeros(Ns,E,D);
    % Find the derivative of the covariance function
    lhyp.x = x; lhyp.y = preC.y;
    [d,Ks] = calcKtest(lhyp,xs);
    for i=1:E
        Xmx = bsxfun(@minus,permute(x,[1,3,2]),permute(xs,[3,1,2])); % N-by-Ns-by-D
        XmxiLam = bsxfun(@times,Xmx,permute(iell2(:,i),[3,2,1]));    % N-by-Ns-by-D
        dKtsdx = bsxfun(@times,XmxiLam,Ks(:,:,i));                        % N-by-Ns-by-D

        df2i = etprod('12',etprod('123',dKtsdx,'513',A(:,:,i),'52'),'142',dKtsdx,'412'); % Ns-by-D
        df2(:,i,:) = bsxfun(@plus,ilsf(:,i)',df2i);
    end
end