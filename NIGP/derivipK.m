function dipKdlsipn = derivipK(lsipn,df2)
% Function to calculate the derivative of ipK w.r.t. the input noise
% hyperparameter: the log standard deviations of the input noise. ipK is a
% diagonal matrix and thus so are its derivatives. This function just returns
% the diagonal elements.
%
% Andrew McHutchon July 2012

[N E D] = size(df2); if E > 1; error('derivipK written for E = 1'); end
df2 = squeeze(df2);                                                     % N-by-D
s2(1,1:D) = exp(2*lsipn);                              % 1-by-D, noise variances
dipKdlsipn = 2*bsxfun(@times, df2, s2);                                 % N-by-D
