function [input, output] = getMeasurementDistributions(sonig, xm, ym)
%getMeasurementDistributions will return two distribution objects for the input and output, given the original measurement.
% This function takes an original measurement, given by the measured input xm and the measured output ym, and turns these into distribution objects. This is done based on the noise intensity
% known by the SONIG object. The resulting input and output distributions can subsequently be implemented in the SONIG object as well, if desired. That function should then be called separately
% though.

% We check the sizes of all the vectors.
if size(xm,1) ~= sonig.dx || size(xm,2) ~= 1
	error(['The getMeasurementDistributions function was called with a vector xm which was not a column vector of size ',num2str(sonig.dx),'.']);
end
if size(ym,1) ~= sonig.dy || size(ym,2) ~= 1
	error(['The getMeasurementDistributions function was called with a vector ym which was not a column vector of size ',num2str(sonig.dy),'.']);
end

% We now set up the corresponding distributions.
input = createDistribution(xm, diag(sonig.hyp.sx.^2));
output = createDistribution(ym, diag(sonig.hyp.sy.^2));

end

