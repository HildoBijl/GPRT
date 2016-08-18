function [nlml, dnlml] = nlmlNIGP(lh, df2, x, y)

% nlmlNIGP - Gaussian process regression, with NIGP input nouse. The
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this is used to fit the hyperparameters.
%
% usage: [nlml dnlml] = nlmlNIGP(lh, df2, x, y)
%
% Inputs:
%   lh         is a struct of log hyperparameters
%     .seard   log lengthscales, log signal std and log output noise std
%     .lsipn   log std input noise
%   df2        is the t-1 iteration of the slopes, fixed for the calculation
%                of the current NIGP fit, as per iterative method
%   x          training inputs matrix, N-by-D
%   y          training targets matrix, N-by-E
%
% Outputs:
%   nlml     value of the negative log marginal likelihood
%   dnlml    struct of partial derivatives of the negative log marginal 
%            likelihood wrt each log hyperparameter
%
% Andrew McHutchon, 11/07/2012

persistent XmX2 oldx

[N, D] = size(x); E = size(y,2);

% Check hyperparameter numbers
if (D+2)*E ~= numel(lh.seard) || D ~= numel(lh.lsipn)
  error(['Error: Wrong number of parameters, lh.seard has %i elements, it should '...
  'have %i; lh.lsipn has %i elements, it should have %i;'],numel(lh.seard),(D+2)*E, ...
  numel(lh.lsipn),D);
end

% If df2 is given use it, else default to zeros
if isempty(df2); df2 = zeros(N,E,D); end;

% XmX2 only needs to be calculated if x changes
if isempty(XmX2) || N ~= size(XmX2,1) || any(size(x)~=size(oldx)) ||any(any(x~=oldx));
    XmX2 = bsxfun(@minus,permute(x,[1,3,2]),permute(x,[3,1,2])).^2; % N-by-N-by-D
    oldx = x;
end
    
% Compute training set covariance matrix, alpha, and chol(alpha)
K = calcK(lh.seard, XmX2);         % N-by-N-by-E, training set covariance matrix
sn2I = bsxfun(@times,permute(exp(2*lh.seard(end,:)),[1,3,2]),eye(N)); % N-by-N-by-E
ipK = df2toipK(lh, df2);           % turn df2 into ipK - df2 is NOT recalculated
Kn = K + sn2I + ipK;               % training set covariance matrix plus noise
R = chol3(Kn); alpha = findAlpha(R,y); Rd = zeros(N,E);
for i=1:E; Rd(:,i) = diag(R(:,:,i)); end  % diag of chol of the covariance
    

% Compute NLML
nlml = sum(sum(0.5*y.*alpha + log(Rd))) + E*0.5*N*log(2*pi);

% NLML Derivatives 
if nargout == 2               % if requested nlml partial derivatives
    dnlml.seard = zeros(D+2,E); dnlml.lsipn = zeros(D,1); 
    dKndlh = zeros(N,N,D+2);
    
    for i=1:E                     % Loop over output dimension, memory efficient
    
        % Find dKndlh
        dKndlh(:,:,1:D+1) = derivK(lh.seard(:,i), K(:,:,i), XmX2); % N-by-N-by-D+1
        dKndlh(:,:,D+2) = bsxfun(@times,2*exp(2*permute(lh.seard(end,i),[3,1,2])),eye(N)); % N-by-N
        dKndlsipn = derivipK(lh.lsipn, df2(:,i,:));                     % N-by-D
    
        % Compute iKn * d/dlh { [K + sn2*I + sip2*df2] } * alpha
        iKn = solve_chol(R(:,:,i),eye(N));                              % N-by-N
        dKndlhalpha = etprod('12',dKndlh,'142',alpha(:,i),'4');       % N-by-D+2
        dKndlsipnalpha = bsxfun(@times,dKndlsipn,alpha(:,i));         % N-by-D+2
        TRiKndKndlh = iKn(:)'*reshape(dKndlh,N^2,D+2);                % 1-by-D+2
        TRiKndKndlsipn = diag(iKn)'*dKndlsipn;                          % 1-by-D
        iKndKndlhalpha = iKn*dKndlhalpha;                             % N-by-D+2
        iKndKndlsipnalpha = iKn*dKndlsipnalpha;                         % N-by-D
    
        % Compute d/dlh { alpha } = - iKn * d/dlh { Kn } * alpha
        dalphadlh = -iKndKndlhalpha; dalphadlsipn = -iKndKndlsipnalpha;
    
        % Compute derivative of NLML
        dnlml.seard(:,i) = 0.5*(dalphadlh'*y(:,i) + TRiKndKndlh');    % D+2-by-1
        dnlml.lsipn = dnlml.lsipn + 0.5*(dalphadlsipn'*y(:,i) + TRiKndKndlsipn'); % D-by-1
    end
end  