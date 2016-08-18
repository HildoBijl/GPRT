function [mu, Sigma] = getGPPrediction(x, Xu, muu, Su, Kuu, lf, lx, eps)
%getGPPrediction Returns the mean and the variance of the GP for a given point x.

% We calculate the necessary matrices.
Ksu = lf^2*exp(-1/2*sum((permute(Xu,[3,2,1]) - repmat(permute(x,[2,3,1]),1,size(Xu,2))).^2./repmat(permute(lx.^2,[3,2,1]),1,size(Xu,2)),3));
Kss = lf^2;

% We calculate the distribution of the output value.
KsuDivKuu = Ksu/(Kuu+eps*eye(size(Kuu)));
mu = KsuDivKuu*muu;
Sigma = Kss - KsuDivKuu*(Kuu-Su)*KsuDivKuu';

end

