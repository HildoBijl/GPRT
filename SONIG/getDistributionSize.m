function [n] = getDistributionSize(dist)
%getDistributionSize will return the size of a distribution. So the number of elements it contains.
% This function returns the size of a distribution object. It also checks the size. So it makes sure that the mean is an n by 1 vector and the covariance matrix is an n by n matrix. It returns n.

% We extract the size and check all the other sizes.
n = size(dist.mean, 1);
if size(dist.mean,2) ~= 1
	error('The distribution provided to the getDistributionSize was not a proper distribution object. Its mean was not a (column) vector.');
end
if size(dist.cov,1) ~= size(dist.cov,2)
	error('The distribution provided to the getDistributionSize was not a proper distribution object. Its covariance matrix was not square.');
end
if size(dist.cov,1) ~= n
	error('The distribution provided to the getDistributionSize was not a proper distribution object. The size of its mean vector did not correspond to the size of its covariance matrix.');
end

end

