function [postMean, postCov, postStd] = makeSonigPrediction(sonig, xt)
%makeSonigPrediction will predict the output value for a given trial input point (or points).
% This function will apply GP regression, predicting the function output value for given trial input points. The input should be the following.
%	sonig: a SONIG object with (hopefully) already measurements implemented into it.
%	xt: a set of trial input points for which to predict the output. It should be a matrix with height sonig.dx and with equal to the number of trial input points nt.
%
% The output of the function is subsequently given by three parameters.
%	postMean: the posterior mean output for the given trial input points in each output direction. This is an nt by sonig.dy vector.
%	postCov: the posterior covariance matrix for the output at the given trial input points. This is an nt by nt by sonig.dy three-dimensional matrix.
%	postStd: the posterior standard deviation for each of the points in each output direction. This is an nt by sonig.dy vector.

% We check the trial input points that we have been given.
if size(xt,1) ~= sonig.dx
	error(['The makeSonigPrediction function was called with points of size ',num2str(size(xt,1)),', while the given SONIG object has points of size ',num2str(sonig.dx),'.']);
end
nt = size(xt,2); % We calculate the number of trial input points.

% Now we start setting up the predictions for each output direction.
postMean = zeros(nt, sonig.dy); % We set up space for the posterior mean.
if nargout >= 2
	postCov = zeros(nt, nt, sonig.dy); % We set up space for the posterior covariance matrices, but only if it was actually asked for.
end
if nargout >= 3
	postStd = zeros(nt, sonig.dy); % We set up space for the posterior standard deviation vectors, but only if it was actually asked for.
end
diff1 = repmat(permute(xt,[3,2,1]),nt,1,1) - repmat(permute(xt,[2,3,1]),1,nt,1); % This is the difference matrix for trialInput with itself, to calculate the matrix Ktt, with the subscript t denoting the trial input points.
diff2 = repmat(permute(xt,[3,2,1]),sonig.nu,1,1) - repmat(permute(sonig.Xu,[2,3,1]),1,nt,1); % This is the difference matrix for trialInput with Xu, to calculate the matrix Kut.
for i = 1:sonig.dy
	% We calculate covariance matrices that we need to add.
	Ktt = sonig.hyp.ly(i)^2*exp(-1/2*sum((diff1./repmat(permute(sonig.hyp.lx(:,i),[2,3,1]),nt,nt,1)).^2,3));
	Kut = sonig.hyp.ly(i)^2*exp(-1/2*sum((diff2./repmat(permute(sonig.hyp.lx(:,i),[2,3,1]),sonig.nu,nt,1)).^2,3));
	Ktu = Kut';
	
	% We now calculate the posterior distributions for the output directions.
	postMean(:,i) = Ktu/sonig.Kuu{i}*sonig.fu{i}.mean;
	if nargout >= 2
		postCov(:,:,i) = Ktt - Ktu/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)/sonig.Kuu{i}*Kut;
	end
	if nargout >= 3
		postStd(:,i) = sqrt(diag(postCov(:,:,i)));
	end
end

end