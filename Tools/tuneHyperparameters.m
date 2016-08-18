function [sfm, lf, lx, mb, logp] = tuneHyperparameters(Xm, fmh)
%tuneHyperparameters This function tunes hyperparameters on a given data set using a simple gradient ascent algorithm.
% The tuneHyperparameters function will tune the hyperparameters of a regular Gaussian process based on an input set xm and an output set ym. Inputs are:
%	Xm: The set of input points. This should be a matrix of size d by n, with d the input vector size and n the number of measurement points.
%	fmh: The set of measured output values. This should be a vector of size n (by 1).
% The output will consist of the tuned hyperparameters and the log-likelihood.
%	sfm: The standard deviation of the output noise.
%	lf: The output length scale.
%	lx: The input length scale, as a vector. This is because we assume Lambda_x is a diagonal matrix, so we only keep track of its diagonal entries. To get the matrix Lambda_x, use diag(lx.^2).
%	mb: The constant value m(x) = \bar{m}.
%	logp: The log-likelihood of the measurements, with the given hyperparameters.

% We derive some dimensions.
dx = size(Xm,1); % This is the input point dimension.
nm = size(Xm,2); % This is the number of measurement points.

% We check the size of ym.
if size(fmh,2) ~= 1
	error('The tuneHyperparameters function was called with invalid input. The ym parameter should be a column vector. However, the given ym parameter did not have length 1 in the horizontal direction.');
end
if size(fmh,1) ~= nm
	error(['The tuneHyperparameters function was called with invalid input. According to the xm parameter, there are ',num2str(nm),' input points, but the ym vector does not have this height.']);
end

% We set up some settings.
numSteps = 50; % This is the number of steps we will do in our gradient ascent.
stepSize = 1; % This is the initial (normalized) step size. In the non-dimensionalized gradient ascent algorithm we use below, this can be seen as a length scale of the optimized parameter, in this case the log-likelihood.
stepSizeFactor = 2; % This is the factor by which we will adjust the step size, in case it turns out to be inadequate.
maxReductions = 100; % This is the maximum number of times in a row which we can reduce the step size. If we'd need this many reductions, something is obviously wrong.

% We define the preliminary hyperparameters.
sfm = 0.1; % This is the initial value of \hat{s}_f.
lf = 1; % This is the initial value of lambda_f.
lx = ones(dx,1); % These are the initial values of lambda_x for each dimension.
hyp = [sfm^2;lf^2;lx.^2]; % We merge the hyperparameters into the hyperparameter array.
newHypDeriv = zeros(2+dx,1); % We already create a storage for the new hyperparameter derivative array. We'll need this soon.

% We calculate the difference matrix for all points a single time.
diff = repmat(permute(Xm,[2,3,1]),[1,nm]) - repmat(permute(Xm,[3,2,1]),[nm,1]);

% Now it's time to start iterating.
for i = 1:numSteps
	% We try to improve the parameters, all the while checking the step size.
	for j = 1:maxReductions
		% We check if we haven't accidentally been decreasing the step size too much.
		if j == maxReductions
			disp('Error: something is wrong with the step size in the hyperparameter optimization scheme.');
		end
		% We calculate new hyperparameters. Or at least, candidates. We will still check them.
		if ~exist('logp','var') % If no logp is defined, this is the first time we are looping. In this case, with no derivative data known yet either, we keep the hyperparameters the same.
			newHyp = hyp;
		else
			newHyp = hyp.*(1 + stepSize*hyp.*hypDeriv); % We apply a non-dimensional update of the hyperparameters. This only works when the parameters are always positive.
		end
		% Now we check the new hyperparameters. If they are good, we will implement them.
		if min(newHyp > 0) % The parameters have to remain positive. If they are not, something is wrong. To be precise, the step size is too big.
			% We extract the new values of the hyperparameters and calculate the new value of logp.
			sfm2 = newHyp(1);
			lf2 = newHyp(2);
			lx2 = newHyp(3:end);
			Kmm = lf2*exp(-1/2*sum(diff.^2./repmat(permute(lx2,[2,3,1]),[nm,nm,1]),3));
			P = Kmm + sfm2*eye(nm);
 			mb = (ones(nm,1)'/P*fmh)/(ones(nm,1)'/P*ones(nm,1)); % This is the (constant) mean function m(x) = \bar{m}.
			newLogp = -nm/2*log(2*pi) - 1/2*logdet(P) - 1/2*(fmh - mb)'/P*(fmh - mb);
			% If this is the first time we are in this loop, or if the new logp is better than the old one, we fully implement the new hyperparameters and recalculate the derivative.
			if ~exist('logp','var') || newLogp >= logp
				% We calculate the new hyperparameter derivative.
				alpha = P\(fmh - mb);
				R = alpha*alpha' - inv(P);
				newHypDeriv(1) = 1/2*trace(R);
				newHypDeriv(2) = 1/(2*lf^2)*trace(R*Kmm);
				for k = 1:dx
					newHypDeriv(2+k) = 1/(4*lx(k)^4)*trace(R*(Kmm.*(diff(:,:,k).^2)));
				end
				% If this is not the first time we run this, we also update the step size, based on how much the (normalized) derivative direction has changed. If the derivative is still in the
				% same direction as earlier, we take a bigger step size. If the derivative is in the opposite direction, we take a smaller step size. And if the derivative is perpendicular to
				% what is used to be, then the step size was perfect and we keep it. For this scheme, we use the dot product.
				if exist('logp','var')
					directionConsistency = ((hypDeriv.*newHyp)'*(newHypDeriv.*newHyp))/norm(hypDeriv.*newHyp)/norm(newHypDeriv.*newHyp);
					stepSize = stepSize*stepSizeFactor^directionConsistency;
				end
				break; % We exit the step-size-reduction loop.
			end
		end
		% If we reach this, it means the hyperparameters we tried were not suitable. In this case, we should reduce the step size and try again. If the step size is small enough, there will
		% always be an improvement of the hyperparameters. (Unless they are fully perfect, which never really occurs.)
		stepSize = stepSize/stepSizeFactor;
	end
	% We update the important parameters.
	hyp = newHyp;
	hypDeriv = newHypDeriv;
	logp = newLogp;
end

% We extract all the parameters.
sfm = sqrt(hyp(1));
lf = sqrt(hyp(2));
lx = sqrt(hyp(3:end));

end

