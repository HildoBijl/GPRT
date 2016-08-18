function [sonig] = createSONIG(lx, sx, ly, sy)
%createSONIG creates a SONIG object which represents a full GP regression algorithm.
% This function will create a SONIG object. This object can be used to apply GP regression using both noisy inputs and noisy outputs.
%
% As input, when creating the SONIG object, all hyperparameters need to be given, and these will remain fixed. They need to be given in this order:
%	lx: the length scales for the input, as a dx by 1 vector. Note that lx = sqrt(diag(Lambda)), with Lambda the matrix of squared length scales often used in literature.
%	sx: the noise scales for the input, as a dx by 1 vector.
%	ly: the noise scales for the output, as a dy by 1 vector. In literature, this scale is often denoted by alpha or sigma_f.
%	sy: the noise scales for the output, as a dy by 1 vector. In literature, this is often denoted by sigma_n or Sigma_n (the matrix) such that sx = sqrt(diag(Sigma_n)).
% Optionally, the hyperparameters can be given as a hyperparameter object.
%
% The output of this function will be a SONIG object. It can then be used by other functions. A SONIG object has the following fields.
%	hyp: the hyperparameters of the GP.
%	dx: the dimension of the input vectors.
%	dy: the dimension of the output vectors.
%	nu: the number of inducing input points used at the moment.
%	Xu: the dx by nu matrix containing the inducing input points.
%	Kuu: the covariance matrix for the inducing input points.
%	Kuui: the inverse of the covariance matrix of the inducing input points.
%	fu: an array of dy distribution objects. Each cell of the array corresponds to one of the outputs of the GP.

% We check the input.
if nargin == 4
	hyp.lx = lx;
	hyp.sx = sx;
	hyp.ly = ly;
	hyp.sy = sy;
elseif nargin == 1
	hyp = lx;
	if ~isfield(hyp,'lx')
		error('The createSONIG function was called with a hyperparameter object which did not contain a field lx.');
	elseif ~isfield(hyp,'sx')
		error('The createSONIG function was called with a hyperparameter object which did not contain a field sx.');
	elseif ~isfield(hyp,'ly')
		error('The createSONIG function was called with a hyperparameter object which did not contain a field ly.');
	elseif ~isfield(hyp,'sy')
		error('The createSONIG function was called with a hyperparameter object which did not contain a field sy.');
	end
else
	error('The createSONIG function was called with the wrong number of parameters.');
end

% We set up the SONIG object, remembering the hyperparameters.
sonig.hyp = hyp;

% We check the sizes of the given vectors.
sonig.dx = size(hyp.lx,1);
sonig.dy = size(hyp.ly,1);
if (size(hyp.lx,2) ~= 1 && size(hyp.lx,2) ~= sonig.dy) || size(hyp.sx,1) ~= sonig.dx || size(hyp.sx,2) ~= 1
	error('The parameters lx and/or sx given to the createSONIG function did not have the proper size. They both need to be nx by 1 arrays.');
end
if size(hyp.ly,2) ~= 1 || size(hyp.sy,1) ~= sonig.dy || size(hyp.sy,2) ~= 1
	error('The parameters ly and/or sy given to the createSONIG function did not have the proper size. They both need to be ny by 1 arrays.');
end
% If the user gave lx as a vector, instead of an array, then we assume that he assumes that the length parameters are the same for every output. So we just repeat the vector into a matrix.
if size(hyp.lx,2) == 1
	hyp.lx = repmat(hyp.lx, 1, sonig.dy);
end

% We set up matrices of inducing input points and their corresponding output distributions for each output.
sonig.nu = 0;
sonig.Xu = zeros(sonig.dx,0);
for i = 1:sonig.dy
	sonig.Kuu{i} = zeros(0,0);
% 	sonig.Kuui{i} = zeros(0,0); % We don't use this anymore, as it results in computational problems in Matlab. Just inverting the matrix every single time seems to be more accurate, without costing too much extra speed.
	sonig.fu{i} = createEmptyDistribution();
end

% And we set some other parameters.
sonig.nm = 0; % This is a counter for the number of measurements implemented.
sonig.valid = 1; % We note that the SONIG is still valid. No numerical inaccuracies have caused problems yet.
sonig.numMeasurementsSkipped = 0; % This is the number of measurements we've already skipped because they would cause the SONIG object to become invalid.

end