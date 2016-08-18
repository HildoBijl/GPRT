function [model fx] = trainNIGP(x,y,Nls,mode,seard,lsipn)
% [lhyp fx] = trainNIGP(x,y,Nls,seard,lsipn)
% Function to train the NIGP model by an iterative method. For each
% iteration the slope values used to refer the input noise to the output
% are fixed and the GP hyperparameters (including the input noise variance)
% are trained. The slope values are then recalculated and used for the next
% iteration. The number of linesearches in each iteration is a parameter
% which can be set and can influence the performance of the algorithm.
% The function keeps track of the best marginal likelihood found
% and returns the associated hyperparameters. If you notice the NLML is
% oscillating try reducing the number of linesearches in each iteration.
%
% Inputs:
%   x       input training data matrix, N-by-D
%   y       training targets matrix, N-by-E
%   Nls     (optional) scalar: (+ve) max total number of linesearches, (-ve) max
%           number of marginal likelihood evaluations. Or a two element vector  
%           where the second element is the max number of linesearches or
%           marginal likelihood evaulations to use *per iteration*. 
%           Decrease this parameter if oscillation ocurrs.
%   mode    (0) derivative of post. mean, (1) include uncertainty in deriv
%   seard   (optional) log initial hyperparameters, excluding input noise
%           variances, for GP kernel, D+2-by-E (for squared exp)
%   lsipn   (optional) initial values for the log input noise standard 
%           deviations, D-by-1
%
% Outputs:
%   model       the trained model returned in a struct:
%      .seard   the trained log GP kernel hyperparameters
%      .lsipn   the trained log input noise standard deviations
%      .df2     the squared slope values at each training point for the
%               'base' GP - figure 1a in the paper, N-by-E-by-D
%      .dipK    the corrective noise terms to be added to the diagonal of K
%
% For more information see the NIPS paper: Gaussian Process Training with
% Input Noise; McHutchon and Rasmussen, 2011. 
% http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon
%
% November 2011, Andrew McHutchon

format compact; addpath util; addpath tprod;
[N D] = size(x); E = size(y,2);

% Set some defaults
if nargin < 3; Nls = -500; mode = 0; end % number of marg. lik. evaulations
if length(Nls) == 1; Nls(2) = ceil(Nls(1)/2); end
if nargin < 4 || isempty(mode); mode = 0; end
options.verbosity = 1; fx = zeros(1,abs(Nls(1)));

if nargin < 6
    % No initial input noise hypers
    lsipn = log(std(x,[],1)/10)';            % D-by-1
    if nargin < 5
        % Also no initial squared exp or output noise hypers
        lell = log(std(x,[],1))';            % D-by-1
        lsf = log(std(y,[],1));              % 1-by-E
        lsn = lsf - log(100);                % 1-by-E, SNR of 100 to start
        seard = [repmat(lell,1,E);lsf;lsn];  % D+2-by-E
    end
end
lh.seard = seard; lh.lsipn = lsipn;

fprintf('Started NIGP training:\r');
bestfx = Inf; nigp.df2 = zeros(N,E,D); nigp.y = y; idx = 1; Lused = 0;
while Lused < abs(Nls(1))
    options.length = sign(Nls(2))*min(abs(Nls(1)) - Lused, abs(Nls(2)));
    
    % Update slopes
    fprintf('calculating slopes...');
    nigp = calcNewdf2(lh,x,nigp,mode); fprintf('done\r');

    % Find next estimate of hyper-parameters
    [lh fX lused] = minimize(lh,'hypCurbNIGP',options,nigp.df2,x,y);
    
    % Book-keeping
    fx(idx) = fX(end); Lused = Lused + lused;
    if fx(idx) < bestfx; bestfx = fx(idx); model = lh; model.df2 = nigp.df2; end
    if idx>1 && abs((fx(idx)-fx(idx-1))/fx(idx)) < 1e-9; break; end   
    idx = idx + 1;
end
fx = fx(1:idx); fprintf('Best NLML: %4.6e\n',bestfx);
model.x = x; model.y = y;
model.dipK = sum(bsxfun(@times,model.df2,permute(exp(2*model.lsipn),[3,2,1])),3);

%%
function nigp = calcNewdf2(lh,x,nigp,mode)

if ~mode; dffunc = @calcdf2; else dffunc = @calcdf2u; end % Function to calculate the slopes

nigp = calcNigp(lh,[],x,nigp);          % With [] => update with new hypers
df2 = dffunc(lh, x, nigp);              % Calculate new squared slopes, df2
nigp = calcNigp(lh,df2,x,nigp);        % With df2 => update with new slopes
%%
function nigp = calcNigp(lh,df2,x,nigp)
% Function to calculate precomputable variables from a change in
% hyperparameters, or df2
N = size(x,1);

% XmX and y
if ~isfield(nigp,'XmX') || size(nigp.XmX,1) ~= N; % Only recalculate if necessary
    nigp.XmX = bsxfun(@minus,permute(x,[1,3,2]),permute(x,[3,1,2])); % N-by-N-by-D
end

% K
nigp.K = calcK(lh.seard, nigp.XmX.^2);

% ipK
if isempty(df2); df2 = nigp.df2; else nigp.df2 = df2; end
ipK = df2toipK(lh, df2);      % turn df2 into ipK - df2 is NOT recalculated

% R and alpha
sn2I = bsxfun(@times,permute(exp(2*lh.seard(end,:)),[1,3,2]),eye(N)); % sn2
Kn = nigp.K + sn2I + ipK;                         % Noisy covariance matrix
nigp.R = chol3(Kn); nigp.alpha = findAlpha(nigp.R,nigp.y); % chol(Kn) & alpha