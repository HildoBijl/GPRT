function [f df] = hypCurbNIGP(lh, df2, x, y, curb)
% wrapper for NIGP training, penalising large SNR and extreme length-scales. Try
% to avoid running into the barrier too hard as it can seriously undermine the
% fit.
%
% Carl Edward Rasmussen, 2011-12-19.
% Adapted for input noise Andrew McHutchon, 12/01/2012

p = 20;                                                          % penalty power
D = size(x,2); xstd = std(x,[],1);

if nargin < 5, curb.snr = 500; curb.ls = 1000; curb.std = xstd; end % set default
maxSNR = log(curb.snr);

% call the marginal likelihood
[f df] = nlmlNIGP(lh,df2,x,y);                            

% penalise too large and too small lengthscales
d = bsxfun(@minus,lh.seard(1:D,:),log(curb.std)'); % log ratio of lengthscale to x std D-by-E
f = f + sum(sum((abs(d)/log(curb.ls)).^p));    
df.seard(1:D,:) = df.seard(1:D,:) + p*(abs(d)/log(curb.ls)).^(p-1)/log(curb.ls).*sign(d);

% signal-to-noise ratio
snr = lh.seard(D+1,:)-lh.seard(D+2,:);  % log signal-to-o/p noise ratio, 1-by-E
f = f + sum((snr/maxSNR).^p);
df.seard(D+1,:) = df.seard(D+1,:) + p*(snr/maxSNR).^(p-1)/maxSNR;
df.seard(D+2,:) = df.seard(D+2,:) - p*(snr/maxSNR).^(p-1)/maxSNR;

snr =  log(xstd)' - lh.lsipn;             % log signal-to-o/p noise ratio, 1-by-D
f = f + sum((snr/maxSNR).^p);
df.lsipn = df.lsipn - p*(snr/maxSNR).^(p-1)/maxSNR;