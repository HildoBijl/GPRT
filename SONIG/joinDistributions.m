function [dist] = joinDistributions(div1,div2,cov)
%joinedDistribution will join two distributions with a given covariance matrix.
% This function will take two distributions, each with a mean and covariance matrix, and join them into one distribution. To do so, we also need to know the covariance between these two
% distributions, which should also be provided to the function. The result will have as mean the concatenated means of the two distributions, and as covariance matrix the concatenated covariance
% matrices of the two distributions (including the covariance between the two distributions).

% We extract some distribution sizes.
d1 = getDistributionSize(div1);
d2 = getDistributionSize(div2);

% We check if a covariance has been given. If not, we assume it to be zero.
if nargin < 3
	cov = zeros(d1, d2);
end

% We check the sizes of the given parameters.
if size(cov,1) ~= d1 || size(cov,2) ~= d2
	error(['The joinDistributions function was called with a covariance matrix of incorrect size. Based on the other two distributions provided, it should have size ',num2str(d1),' by ',num2str(d2),', but its size was ',num2str(size(cov,1)),' by ',num2str(size(cov,2)),'.']);
end

% We set up the requested distribution.
dist = createDistribution([div1.mean;div2.mean], [div1.cov,cov;cov',div2.cov]);

end

