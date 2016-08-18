function [sonig, newInput, newOutput, newJointDist] = implementMeasurement(sonig, input, output, jointDist)
%implementMeasurement Will implement a measurement in the SONIG algorithm, updating the SONIG prediction and returning posterior distributions of the measurement.
% This function implements an input-output measurement into the SONIG algorithm. The function input should consist of three parameters.
%	A SONIG object, which will use the information from the measurement to refine its distribution(s) of inducing input points.
%	An input distribution. This should be a (Gaussian) distribution object, reflecting both the mean and the covariance of the input (position) of the measurement.
%	An output distribution. This should be a (Gaussian) distribution object, reflecting both the mean and the covariance of the output (value) of the measurement.
%	Optional: a joint distribution of the input and the output. This is a way to take into account any possible covariance between the input and the output. If this parameter is not given, the
%	covariance is assumed to be zero.
%
% The function then returns the exact same three objects, but then with posterior distributions. In addition, it also returns the joint distribution between the input and the output. So it returns:
%	A SONIG object whose inducing input point distribution(s) have incorporated the new measurement.
%	A posterior input distribution of what, given the data in the SONIG object, we expect the actual measurement point to have been.
%	A posterior output distribution of what, given the data in the SONIG object, we expect the actual output value to have been.
%	A posterior joint distribution, representing the joint distribution of the input and output. (It first has the input and then the output.) This doesn't only contain the information of the
%		previous two distributions, but it also includes the covariance between the input and the output.

% We first check that the distribution we've been given are of the proper size.
dx = getDistributionSize(input);
if sonig.dx ~= dx
	error('The implementMeasurement function was called with an input distribution whose size was not corresponding to the input size of the SONIG object.');
end
dy = getDistributionSize(output);
if sonig.dy ~= dy
	error('The implementMeasurement function was called with an output distribution whose size was not corresponding to the output size of the SONIG object.');
end
% We now check the joint distribution. If it has been given, we check it. Otherwise, we set it up ourselves.
if nargin < 4
	jointDist = joinDistributions(input, output);
else
	dxy = getDistributionSize(jointDist);
	if dx + dy ~= dxy
		error('The implementMeasurement function was called with a joint distribution whose size was not corresponding to the joint sizes of the input and output.');
	end
end
% And we extract the covariance from the joint distribution.
covXY = jointDist.cov(1:sonig.dx,sonig.dx+1:end);
covYX = covXY';
realOutputCov = output.cov - covYX/input.cov*covXY;

% We check if we've still got a valid SONIG object. If not, we don't do anything anymore.
if sonig.valid == 0
	newInput = input;
	newOutput = output;
	newJointDist = jointDist;
	return;
end
	
% The first thing which we should now do is calculate the posterior distribution of the input point.
xl = input.mean; % This is the point we linearize about.
smallestLengthScales = min(sonig.hyp.lx,[],2); % We calculate the smallest length scales for each input dimension. We'll use those to normalize distances whenever required.
inputMean = input.mean; % We extract the input mean. We will be updating this a few times to calculate the posterior input mean.
inputCov = input.cov; % We extract the input covariance. We will be updating this a few times to calculate the posterior input covariance.
counter = 0;
maxCounterIterations = 30;
while counter < maxCounterIterations && (exist('prevxl','var') == 0 || sum(((xl-prevxl)./smallestLengthScales).^2) > 0.001) % We continue iterating until the change between xl and the previous xl (when normalized) is below a certain threshold. We also have a security check on the maximum number of iterations.
	% We do an iteration to update the posterior input for a given linearization point.
	counter = counter + 1;
	diff = permute(sonig.Xu,[3,2,1]) - repmat(permute(xl,[2,3,1]),[1,sonig.nu,1]);
	mux = zeros(sonig.dy, 1); % We prepare a vector for the mean of the output at the linearization point.
	dmux = zeros(sonig.dy, sonig.dx); % We prepare a matrix containing the derivatives of dmu_+ for each output dimension.
	Sxx = zeros(sonig.dy, sonig.dy); % We prepare a matrix for Sigma_++ which is the covariance of the output at the linearization point. By assumption, it's a diagonal matrix.
	for i = 1:sonig.dy
		% We examine a single output dimension.
		lengthDivisor = repmat(permute(sonig.hyp.lx(:,i),[3,2,1]),[1,sonig.nu,1]);
		diffNormalized = diff./lengthDivisor;
		diffNormalized2 = diffNormalized./lengthDivisor;
		Kxx = sonig.hyp.ly(i)^2;
		Kxu = sonig.hyp.ly(i)^2*exp(-1/2*sum(diffNormalized.^2,3));
		Kux = Kxu';
		dKxu = repmat(Kxu,[1,1,sonig.dx]).*diffNormalized2;
% 		dKux = permute(dKxu,[2,1,3]); % This line has been deemed unnecessary.
		mux(i) = Kxu/sonig.Kuu{i}*sonig.fu{i}.mean;
		dmux(i,:) = (permute(dKxu,[3,2,1])/sonig.Kuu{i}*sonig.fu{i}.mean)';
		Sxx(i,i) = Kxx - Kxu/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}*Kux;
	end
	% We calculate the new mean and covariance of the prior distribution.
	inputCov = eye(sonig.dx)/(dmux'/(Sxx + realOutputCov)*dmux + eye(sonig.dx)/input.cov); % We use I/(...) instead of inv(...) because for some strange reason this is more accurate in Matlab.
	inputMean = input.mean + inputCov*(dmux'/(Sxx + realOutputCov)*(dmux*(xl - input.mean) + (output.mean + covYX/input.cov*(xl - input.mean) - mux)));
	% We update the linearization point.
	prevxl = xl;
	xl = inputMean;
end
% We check if the counter reached its limit.
if counter >= maxCounterIterations
	disp(['There was a problem implementing measurement ',num2str(sonig.nm+1),' because the posterior input point distribution could not be found. The algorithm did not converge to a proper linearization point. The measurement is being skipped.']);
	sonig.numMeasurementsSkipped = sonig.numMeasurementsSkipped + 1;
	newInput = input;
	newOutput = output;
	newJointDist = jointDist;
	return;
end
% We have stabilized at a suitable linearization point. We set the input mean and covariance.
newInput = createDistribution(inputMean, inputCov);
adjustedOutput = createDistribution(output.mean + covYX/input.cov*(newInput.mean - input.mean), realOutputCov); % This is a compensation for when there is a prior covariance between the input and the output measurement distributions. It's an approximation, not described in the paper, and I'm not sure whether there might be a more accurate version of this. Something to look into.

% At this point, we can also check if we automatically need to add an inducing input point. We do this if the minimum (normalized) distance from the input point to any inducing input point is less
% than a given threshold. So if no inducing input points are nearby, we add a new one at the position of the posterior mean of the input.
if isfield(sonig, 'addIIPDistance')
	if sonig.nu == 0 || min(sum(((repmat(inputMean, [1,sonig.nu]) - sonig.Xu)./repmat(smallestLengthScales, [1,sonig.nu])).^2,1),[],2) > sonig.addIIPDistance^2
		sonig = addInducingInputPoint(sonig, inputMean);
	end
end

% Next, we need to calculate the posterior distributions. For this, we need to calculate derivatives of various parameters. We set up storage space for that first.
Kxx = zeros(1,1,1,1,sonig.dy);
Kux = zeros(sonig.nu,1,1,1,sonig.dy);
Kxu = zeros(1,sonig.nu,1,1,sonig.dy);
dKux = zeros(sonig.nu,1,sonig.dx,1,sonig.dy);
dKxu = zeros(1,sonig.nu,sonig.dx,1,sonig.dy);
d2Kux = zeros(sonig.nu,1,sonig.dx,sonig.dx,sonig.dy);
d2Kxu = zeros(1,sonig.nu,sonig.dx,sonig.dx,sonig.dy);
Sxx = zeros(1,1,1,1,sonig.dy);
dSxx = zeros(1,1,sonig.dx,1,sonig.dy);
d2Sxx = zeros(1,1,sonig.dx,sonig.dx,sonig.dy);
Q = zeros(1,1,1,1,sonig.dy);
dQ = zeros(1,1,sonig.dx,1,sonig.dy);
d2Q = zeros(1,1,sonig.dx,sonig.dx,sonig.dy);
% Qi = zeros(1,1,sonig.dy); % This line has been deemed unnecessary.
dQi = zeros(1,1,sonig.dx,1,sonig.dy);
d2Qi = zeros(1,1,sonig.dx,sonig.dx,sonig.dy);

% Next, we start calculating the derivatives of all important parameters. For this, we use the definition of the squared exponential covariance function. Yes, I know the notation is a mess with
% all the different matrix dimensions, subscripts and everything. If you can think of a better way to set up this notation, do let me know.
diff = permute(sonig.Xu,[3,2,1]) - repmat(permute(newInput.mean,[2,3,1]),[1,sonig.nu,1]);
for i = 1:sonig.dy
	lengthDivisor = repmat(permute(sonig.hyp.lx(:,i),[3,2,1]),[1,sonig.nu,1]);
	diffNormalized = diff./lengthDivisor;
	diffNormalized2 = diffNormalized./lengthDivisor;
	Kxx(:,:,:,:,i) = sonig.hyp.ly(i)^2;
	Kxu(:,:,:,:,i) = sonig.hyp.ly(i)^2*exp(-1/2*sum(diffNormalized.^2,3));
	Kux(:,:,:,:,i) = Kxu(:,:,:,:,i)';
	dKxu(:,:,:,:,i) = repmat(Kxu(:,:,:,:,i),[1,1,sonig.dx]).*diffNormalized2;
	dKux(:,:,:,:,i) = permute(dKxu(:,:,:,:,i),[2,1,3,4,5]);
	d2Kxu(:,:,:,:,i) = repmat(Kxu(:,:,:,:,i),[1,1,sonig.dx,sonig.dx]).*(mmat(diffNormalized2, permute(diffNormalized2, [1,2,4,3]), [3,4]) - repmat(permute(diag(1./sonig.hyp.lx(:,i).^2),[3,4,1,2]),[1,sonig.nu,1,1]));
	d2Kux(:,:,:,:,i) = permute(d2Kxu(:,:,:,:,i), [2,1,3,4]);
	Sxx(:,:,:,:,i) = Kxx(:,:,:,:,i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}*Kux(:,:,:,:,i);
	dSxx(:,:,:,:,i) = mmat(repmat(-2*Kxu(:,:,:,:,i)/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}, [1,1,sonig.dx]), dKux(:,:,:,:,i));
	d2Sxx(:,:,:,:,i) = -2*permute(permute(dKxu(:,:,:,:,i),[3,2,1])/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}*permute(dKux(:,:,:,:,i),[1,3,2]), [3,4,1,2]) - 2*mmat(repmat(Kxu(:,:,:,:,i)/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}, [1,1,sonig.dx,sonig.dx]), d2Kux(:,:,:,:,i));
	Q(:,:,:,:,i) = Sxx(:,:,:,:,i) + realOutputCov(i,i);
	dQ(:,:,:,:,i) = dSxx(:,:,:,:,i);
	d2Q(:,:,:,:,i) = d2Sxx(:,:,:,:,i);
% 	Qi(:,:,:,:,i) = 1/Q(:,:,:,:,i); % This line has been deemed unnecessary.
	dQi(:,:,:,:,i) = -(1/Q(:,:,:,:,i)^2)*dQ(:,:,:,:,i);
	d2Qi(:,:,:,:,i) = 2*(1/Q(:,:,:,:,i)^3)*mmat(dQ(:,:,:,:,i), permute(dQ(:,:,:,:,i),[1,2,4,3]), [3,4]) - (1/Q(:,:,:,:,i)^2)*d2Q(:,:,:,:,i);
end

% The next step is to calculate the mean and covariance, and their derivatives, of the posterior output. I haven't written down these derivations in any paper, so I hope I did them correctly.
mux = zeros(sonig.dy,1,1,1);
dmux = zeros(sonig.dy,1,sonig.dx,1);
d2mux = zeros(sonig.dy,1,sonig.dx,sonig.dx);
Sxxp = zeros(sonig.dy,sonig.dy,1,1);
dSxxp = zeros(sonig.dy,sonig.dy,sonig.dx,1);
d2Sxxp = zeros(sonig.dy,sonig.dy,sonig.dx,sonig.dx);
for i = 1:sonig.dy
	mux(i,:,:,:) = adjustedOutput.cov(i,i)/Q(:,:,:,:,i)*(Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean) + Sxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.mean(i);
	dmux(i,:,:,:) = adjustedOutput.cov(i,i)*dQi(:,:,:,:,i)*(Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean) + adjustedOutput.cov(i,i)/Q(:,:,:,:,i)*mmat(dKxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx])) + dSxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.mean(i) + Sxx(:,:,:,:,i)*dQi(:,:,:,:,i)*adjustedOutput.mean(i) + Sxx(:,:,:,:,i)/Q(:,:,:,:,i)*permute(covYX(i,:)/input.cov, [1,3,2]);
	term1 = adjustedOutput.cov(i,i)*d2Qi(:,:,:,:,i)*(Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean);
	term2 = dmmat(adjustedOutput.cov(i,i)*dQi(:,:,:,:,i), permute(mmat(dKxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx])), [1,2,4,3]));
	term3 = permute(term2, [1,2,4,3]);
	term4 = adjustedOutput.cov(i,i)/Q(:,:,:,:,i)*mmat(d2Kxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx,sonig.dx]));
	term5 = d2Sxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.mean(i);
	term6 = dmmat(dSxx(:,:,:,:,i), permute(dQi(:,:,:,:,i), [1,2,4,3]))*adjustedOutput.mean(i);
	term7 = permute(term6, [1,2,4,3]);
	term8 = Sxx(:,:,:,:,i)*d2Qi(:,:,:,:,i)*adjustedOutput.mean(i);
	term9 = mmat((dSxx(:,:,:,:,i)/Q(:,:,:,:,i)),permute(covYX(i,:)/input.cov, [1,4,3,2]),[3,4]);
	term10 = mmat((Sxx(:,:,:,:,i)*dQi(:,:,:,:,i)),permute(covYX(i,:)/input.cov, [1,4,3,2]),[3,4]);
	term11 = permute(term9, [1,2,4,3]);
	term12 = permute(term10, [1,2,4,3]);
	d2mux(i,:,:,:) = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12;
	Sxxp(i,i,:,:) = Sxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.cov(i,i);
	dSxxp(i,i,:,:) = dSxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.cov(i,i) + Sxx(:,:,:,:,i)*dQi(:,:,:,:,i)*adjustedOutput.cov(i,i);
	term1 = d2Sxx(:,:,:,:,i)/Q(:,:,:,:,i)*adjustedOutput.cov(i,i);
	term2 = dmmat(dSxx(:,:,:,:,i), permute(dQi(:,:,:,:,i), [1,2,4,3]))*adjustedOutput.cov(i,i);
	term3 = permute(term2, [1,2,4,3]);
	term4 = Sxx(:,:,:,:,i)*d2Qi(:,:,:,:,i)*adjustedOutput.cov(i,i);
	d2Sxxp(i,i,:,:) = term1 + term2 + term3 + term4;
end
% Now that we have these derivatives, we can calculate the posterior distribution of the output.
outputMean = mux + (1/2)*mtrace(mmat(d2mux,repmat(permute(newInput.cov,[3,4,1,2]),[sonig.dy,1,1,1]),[3,4]),[3,4]);
outputCov = Sxxp + (1/2)*mtrace(mmat(d2Sxxp,repmat(permute(newInput.cov,[3,4,1,2]),[sonig.dy,sonig.dy,1,1]),[3,4]),[3,4]) + dmmat(permute(dmux,[1,2,4,3]),dmmat(permute(newInput.cov,[3,4,1,2]),permute(dmux,[2,1,3,4])));
jointCov = permute(dmmat(permute(dmux,[1,2,4,3]),permute(newInput.cov,[3,4,1,2])),[1,4,2,3]);
newOutput = createDistribution(outputMean, outputCov);
newJointDist = createDistribution([inputMean;outputMean], [inputCov,jointCov';jointCov,outputCov]);

% Finally, we will update the distribution of the inducing input points.
for i = 1:sonig.dy
	% We calculate derivatives of the mean and covariance functions.
	dmuu = mmat(repmat(sonig.fu{i}.cov/sonig.Kuu{i}, [1,1,sonig.dx]), dKux(:,:,:,:,i)/Q(:,:,:,:,i)*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean) + mmat(repmat(Kux(:,:,:,:,i),[1,1,sonig.dx]), dQi(:,:,:,:,i))*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean) + mmat(repmat(Kux(:,:,:,:,i)/Q(:,:,:,:,i), [1,1,sonig.dx]), permute(covYX(i,:)/input.cov, [1,3,2]) - mmat(dKxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx]))));
	inter1 = mmat(dKux(:,:,:,:,i), repmat(permute(dQi(:,:,:,:,i), [1,2,4,3]), [sonig.nu,1,1,1]), [3,4]); % This is dKup*dQi, necessary for terms 2 and 4.
	inter2 = permute(permute(covYX(i,:)/input.cov, [1,3,2]) - mmat(dKxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx])), [1,2,4,3]); % This is dKpu/Kuu*muu, necessary for terms 3, 6, 7, 8 and 9.
	term1 = d2Kux(:,:,:,:,i)/Q(:,:,:,:,i)*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean);
	term2 = inter1*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean);
	term3 = mmat(dKux(:,:,:,:,i), Q(:,:,:,:,i)\repmat(inter2, [sonig.nu,1,1,1]), [3,4]);
	term4 = permute(term2, [1,2,4,3]);
	term5 = mmat(repmat(Kux(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]), d2Qi(:,:,:,:,i))*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean);
	term6 = mmat(repmat(Kux(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]), mmat(dQi(:,:,:,:,i), inter2, [3,4]));
	term7 = permute(term3, [1,2,4,3]);
	term8 = permute(term6, [1,2,4,3]);
	term9 = -mmat(repmat(Kux(:,:,:,:,i)/Q(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]), mmat(d2Kxu(:,:,:,:,i), repmat(sonig.Kuu{i}\sonig.fu{i}.mean, [1,1,sonig.dx,sonig.dx])));
	d2muu = mmat(repmat(sonig.fu{i}.cov/sonig.Kuu{i}, [1,1,sonig.dx,sonig.dx]), term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9);
% 	dSu = -mmat(repmat(sonig.fu{i}.cov/sonig.Kuu{i}, [1,1,sonig.dx]), mmat(mmat(dKux(:,:,:,:,i), repmat(Q(:,:,:,:,i)\Kxu(:,:,:,:,i), [1,1,sonig.dx])) + mmat(mmat(repmat(Kux(:,:,:,:,i), [1,1,sonig.dx]), dQi(:,:,:,:,i)), repmat(Kxu(:,:,:,:,i), [1,1,sonig.dx])) + mmat(repmat(Kux(:,:,:,:,i)/Q(:,:,:,:,i), [1,1,sonig.dx]), dKxu(:,:,:,:,i)), repmat(sonig.Kuu{i}\sonig.fu{i}.cov, [1,1,sonig.dx]))); % This line has been deemed unnecessary.
	term1 = (1/Q(:,:,:,:,i))*mmat(d2Kux(:,:,:,:,i), repmat(Kxu(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]));
	term2 = mmat(dmmat(dKux(:,:,:,:,i), permute(dQi(:,:,:,:,i), [1,2,4,3])), repmat(Kxu(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]));
	term3 = (1/Q(:,:,:,:,i))*dmmat(dKux(:,:,:,:,i), permute(dKxu(:,:,:,:,i), [1,2,4,3]));
	term4 = permute(term2, [1,2,4,3]);
	term5 = mmat(mmat(repmat(Kux(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]), d2Qi(:,:,:,:,i)), repmat(Kxu(:,:,:,:,i), [1,1,sonig.dx,sonig.dx]));
	term6 = permute(term4, [2,1,3,4]);
	term7 = permute(term3, [1,2,4,3]);
	term8 = permute(term6, [1,2,4,3]);
	term9 = permute(term1, [2,1,3,4]);
	d2Su = -mmat(repmat(sonig.fu{i}.cov/sonig.Kuu{i}, [1,1,sonig.dx,sonig.dx]), mmat(term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9, repmat(sonig.Kuu{i}\sonig.fu{i}.cov, [1,1,sonig.dx,sonig.dx])));
	% We calculate the new covariance matrix after this prediction and check if it's still valid. If an eigenvalue has turned negative due to numerical issues, we set the valid flag to zero to indicate this.
	newMean = sonig.fu{i}.mean + sonig.fu{i}.cov/sonig.Kuu{i}*Kux(:,:,:,:,i)/Q(:,:,:,:,i)*(adjustedOutput.mean(i) - Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.mean) + (1/2)*mtrace(mmat(d2muu, repmat(permute(newInput.cov, [3,4,1,2]), [sonig.nu,1,1,1]), [3,4]),[3,4]);
	newCov = sonig.fu{i}.cov - sonig.fu{i}.cov/sonig.Kuu{i}*Kux(:,:,:,:,i)/Q(:,:,:,:,i)*Kxu(:,:,:,:,i)/sonig.Kuu{i}*sonig.fu{i}.cov + mtrace(mmat(dmmat(dmuu, permute(dmuu, [2,1,4,3])) + 1/2*d2Su, repmat(permute(newInput.cov,[3,4,1,2]), [sonig.nu,sonig.nu,1,1]), [3,4]), [3,4]);
	minEigenvalue = min(eig(newCov));
	if minEigenvalue < 0
		disp(['Measurement ',num2str(sonig.nm+1),' could not be implemented properly. Due to numerical reasons, its implementation would result in invalid predictions. So it has been ignored.']);
		sonig.numMeasurementsSkipped = sonig.numMeasurementsSkipped + 1;
		if sonig.numMeasurementsSkipped >= 10
			disp('There have been too many invalid measurements. Probably the SONIG results have become invalid.');
			sonig.valid = 0;
			break;
		end
	else
		% We now make the prediction of the mean and the covariance matrix.
		sonig.fu{i}.mean = newMean;
		sonig.fu{i}.cov = newCov;
	end
end

% We have finished implementing a measurement. We increase the counter.
sonig.nm = sonig.nm + 1;

end