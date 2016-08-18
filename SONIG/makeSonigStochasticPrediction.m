function [fDist] = makeSonigStochasticPrediction(sonig, xDist)
%makeSonigStochasticPrediction will predict the output value for a given stochastic trial input point. (Only one point is allowed.)
% This function will apply GP regression, predicting the function output value for given trial input points. The input should be the following.
%	sonig: a SONIG object with (hopefully) already measurements implemented into it.
%	xDist: a trial input point distribution for which to predict the output. It should just be a distribution of size sonig.dx.
%
% The output of the function is subsequently given by a single parameter.
%	fDist: a distribution object giving the posterior distribution of the output. This distribution is of size sonig.dy.

% We check the trial input point that we have been given.
dx = getDistributionSize(xDist);
if dx ~= sonig.dx
	error(['The makeSonigStochasticPrediction function was called with a point of size ',num2str(dx),', while the given SONIG object has points of size ',num2str(sonig.dx),'.']);
end

% We set up some helpful vectors/matrices.
q = zeros(sonig.nu,sonig.dy);
Q = zeros(sonig.nu,sonig.nu,sonig.dy,sonig.dy);
diff = sonig.Xu - repmat(xDist.mean, [1,sonig.nu]); % This is the difference matrix for the trialInput mean with Xu.
diffNormalized = repmat(diff,[1,1,sonig.dy])./repmat(permute(sonig.hyp.lx.^2, [1,3,2]),[1,sonig.nu,1]);
for i = 1:sonig.dy
	q(:,i) = sonig.hyp.ly(i)^2/sqrt(det(xDist.cov)*det(eye(sonig.dx)/xDist.cov + diag(1./sonig.hyp.lx(:,i).^2)))*exp(-1/2*sum((diff'/(diag(sonig.hyp.lx(:,i).^2) + xDist.cov)).*diff', 2));
	for j = 1:sonig.dy
		xn = repmat(permute(diffNormalized(:,:,i),[1,3,2,4]),[1,1,1,sonig.nu]) + repmat(permute(diffNormalized(:,:,j),[1,3,4,2]),[1,1,sonig.nu,1]);
		Q(:,:,i,j) = sonig.hyp.ly(i)^2*sonig.hyp.ly(j)^2/sqrt(det(xDist.cov)*det(eye(sonig.dx)/xDist.cov + diag(1./sonig.hyp.lx(:,i).^2) + diag(1./sonig.hyp.lx(:,j).^2)))*(exp(-1/2*sum(diffNormalized(:,:,i).*diff,1))'*exp(-1/2*sum(diffNormalized(:,:,j).*diff,1))).*permute(exp(1/2*mmat(mmat(permute(xn,[2,1,3,4]), repmat(eye(sonig.dx)/(eye(sonig.dx)/xDist.cov + diag(1./sonig.hyp.lx(:,i).^2) + diag(1./sonig.hyp.lx(:,j).^2)), [1,1,sonig.nu,sonig.nu])), xn)),[3,4,1,2]); % We use I/(...) instead of inv(...) because for some strange reason this is more accurate in Matlab.
	end
end

% We calculate the posterior distribution of the output.
fMean = zeros(sonig.dy,1);
fCov = zeros(sonig.dy,sonig.dy);
for i = 1:sonig.dy
	fMean(i,:) = q(:,i)'/sonig.Kuu{i}*sonig.fu{i}.mean;
	fCov(i,i) = sonig.hyp.ly(i)^2 - trace(sonig.Kuu{i}\(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}*Q(:,:,i,i));
end
for i = 1:sonig.dy
	for j = 1:sonig.dy
		fCov(i,j) = fCov(i,j) + sonig.fu{i}.mean'/sonig.Kuu{i}*Q(:,:,i,j)/sonig.Kuu{j}*sonig.fu{j}.mean - fMean(i,:)*fMean(j,:);
	end
end
% We walk through the diagonal elements of the covariance matrix to make sure they're not negative. This is an extra check, because sometimes Matlab has numerical issues which cause diagonal
% elements of the fCov matrix to be negative. Yes, it's a crude fix, but it helps prevent a few of the problems.
for i = 1:sonig.dy
	fCov(i,i) = max(fCov(i,i),1e-16);
end
fDist = createDistribution(fMean, fCov);

end