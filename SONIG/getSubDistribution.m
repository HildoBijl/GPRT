function [subDist] = getSubDistribution(dist, indices)
%getSubDistribution returns a subdistribution of the given distribution. It gives the distribution consisting of the elements with the given indices.

subDist = createDistribution(dist.mean(indices), dist.cov(indices, indices));

end

