function [M S] = gpm(model, m)
% Function to make NIGP predictions for Gaussian distributed test points.
% This is the slow and ugly version which loops over the Ns test points and
% E output diemsnions. Unfortunately there isn't a fast pretty version of 
% it yet.
%
% Andrew McHutchon

input = model.x; target = model.y; alpha = model.alpha;
s = diag(exp(2*model.lsipn));     % input cov matrix is diag of input noise
[N D] = size(input);                                  % Training dimensions
[Ns Ds] = size(m);                                        % Test dimensions
if Ds ~= D; error('Test point(s) supplied wrongly'); end
E = size(target,2);

M = zeros(Ns,E); S = zeros(Ns,E);
iella = exp(-model.seard(1:D,:))'; % E-by-D
sf2a = exp(2*model.seard(D+1,:));

for n = 1:Ns
    inp = bsxfun(@minus,input,m(n,:));

    for i=1:E
        L = diag(iella(i,:));                                      % D-by-D
        iLam = L.^2;                                               % D-by-D
        in = inp*L;         % N-by-D, training data minus m divided by ells
        B = L*s*L+eye(D);                                          % D-by-D
        iB = B\eye(D);      % D-by-D
        t = in*iB;          % in*inv(B) - N-by-D, O(D^3 x E)
        l = exp(-sum(in.*t,2)/2);  % N-by-1
        lb = l.*alpha(:,i);    % N-by-1
        c = exp(2*model.seard(D+1,i))/sqrt(det(B));
  
        M(n,i) = c*sum(lb);
        
        k = 2*model.seard(D+1,i)-sum(in.*in,2)/2;       % N-by-1, O(ND x E)
        ii = bsxfun(@times,inp,iella(i,:).^2);          % N-by-D, O(ND x E)
        
        C = 2*s*iLam+eye(D);                     % D-by-D, O(D^2 x E/2 x E)
        t = 1/sqrt(det(C));                        % scalar, O(D^3 x E^2/2)
        Qexp = exp(bsxfun(@plus,k,k')+maha(ii,-ii,C\s/2)); % N-by-N, O(N^2 xE^2/2) or O(D^3 xE^2/2)
        
        A = alpha(:,i)*alpha(:,i)';                 % N-by-N, O(N^2 xE^2/2)
        A = A - solve_chol(model.R(:,:,i),eye(N)); % incorporate model uncertainty
        A = A.*Qexp;                                % N-by-N, O(N^2 xE^2/2)
        
        S(n,i) = sf2a(i) + t*sum(sum(A)) - M(n,i)^2;        % O(N^2 xE^2/2)
    end
end