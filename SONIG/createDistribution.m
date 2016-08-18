function [dist] = createDistribution(mean, cov)
%createDistribution sets up a distribution object with the given mean vector and covariance matrix.

% We check the sizes of the given parameters.
d = size(mean,1);
if size(mean,2) ~= 1
	error('The createDistribution function was called with a mean vector which was not a column vector.');
end
if size(cov,1) ~= size(cov,2)
	error('The createDistribution function was called with a covariance matrix which was not square.');
end
if size(cov,1) ~= d
	error('The createDistribution function was called with a mean vector and a covariance matrix whose sizes did not correspond.');
end

% We set up a distribution object and return it.
dist.mean = mean;
dist.cov = cov;

end

