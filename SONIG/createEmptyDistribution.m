function [dist] = createEmptyDistribution()
%createEmptyDistribution creates an empty distribution object.
% This function creates an empty distribution object. It can subsequently be filled by adding elements to the distribution.

dist = createDistribution(zeros(0,1), zeros(0,0));

end

