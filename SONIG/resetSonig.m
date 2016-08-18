function [sonig] = resetSonig(sonig)
%resetSonig will reset the distribution of the SONIG to its prior distribution. So fu (for every output) is set back to the distribution with zero mean and Kuu covariance matrix.

% We walk through the output directions and reset each one of them.
for i = 1:sonig.dy
	sonig.fu{i} = createDistribution(zeros(sonig.nu,1), sonig.Kuu{i});
end

end

