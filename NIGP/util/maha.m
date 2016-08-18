function K = maha(a, b, Q)                         

% Squared Mahalanobis distance (a-b)*Q*(a-b)'; vectors are row-vectors
%
% a, b  matrices containing n length d row vectors, d by n
% Q     weight matrix, d by d, default eye(d)
% K     squared distances, n by n
%
% Copyright (C) 2008-2009 by Marc Deisenroth and Carl Edward Rasmussen, 2009-10-17

if nargin == 2                                                   % assume unit Q
  K = bsxfun(@plus,sum(a.*a,2),sum(b.*b,2)')-2*a*b';
else
  aQ = a*Q; K = bsxfun(@plus,sum(aQ.*a,2),sum(b*Q.*b,2)')-2*aQ*b';
end