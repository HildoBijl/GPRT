function [sonig] = addInducingInputPoint(sonig, xu)
%addInducingInputPoint will add an inducing input point to the given SONIG object.
% This function takes a SONIG object and adds an inducing input point to it. It then updates all the matrices to smoothly incorporate this new inducing input point. The point should be given as a
% vector. It's also possible to add multiple points at the same time. To do so, append the (vertical) vectors to be added horizontally into a matrix.

% We check the inducing input points that we have been given.
if size(xu,1) ~= sonig.dx
	error(['The addInducingInputPoint function was called with points of size ',num2str(size(xu,1)),', while the given SONIG object has points of size ',num2str(sonig.dx),'.']);
end
np = size(xu,2); % We calculate the number of points to be added.

% We walk through the output dimensions to update the Kuu, Kuui and fu distribution for each of them.
diff1 = repmat(permute(xu,[3,2,1]),np,1,1) - repmat(permute(xu,[2,3,1]),1,np,1); % This is the difference matrix for xu with itself, to calculate the addition Kpp, with the subscript p denoting the new points.
diff2 = repmat(permute(xu,[3,2,1]),sonig.nu,1,1) - repmat(permute(sonig.Xu,[2,3,1]),1,np,1); % This is the difference matrix for xu with Xu, to calculate the addition Kup.
for i = 1:sonig.dy
	% We calculate covariance matrices that we need to add.
	Kpp = sonig.hyp.ly(i)^2*exp(-1/2*sum((diff1./repmat(permute(sonig.hyp.lx(:,i),[2,3,1]),np,np,1)).^2,3));
	Kup = sonig.hyp.ly(i)^2*exp(-1/2*sum((diff2./repmat(permute(sonig.hyp.lx(:,i),[2,3,1]),sonig.nu,np,1)).^2,3));
	Kpu = Kup';
	
	% We now update the distribution for the inducing input points.
	sonig.fu{i}.mean = [sonig.fu{i}.mean; Kpu/sonig.Kuu{i}*sonig.fu{i}.mean];
	sonig.fu{i}.cov = [sonig.fu{i}.cov, sonig.fu{i}.cov*(sonig.Kuu{i}\Kup); Kpu/sonig.Kuu{i}*sonig.fu{i}.cov, Kpp - Kpu/sonig.Kuu{i}*(sonig.Kuu{i} - sonig.fu{i}.cov)*(sonig.Kuu{i}\Kup)];
	
	% Finally we update the Kuu matrices, with first the inverse matrix (in a computationally efficient way) and then the regular matrix.
% 	sonig.Kuui{i} = [sonig.Kuui{i} + sonig.Kuui{i}*Kup/(Kpp - Kpu*sonig.Kuui{i}*Kup)*Kpu*sonig.Kuui{i}, -sonig.Kuui{i}*Kup/(Kpp - Kpu*sonig.Kuui{i}*Kup); -(Kpp - Kpu*sonig.Kuui{i}*Kup)\Kpu*sonig.Kuui{i}, inv(Kpp - Kpu*sonig.Kuui{i}*Kup)]; % We don't use this anymore, as it results in computational problems in Matlab. Just inverting the matrix every single time seems to be more accurate, without costing too much extra speed.
	sonig.Kuu{i} = [sonig.Kuu{i}, Kup; Kpu, Kpp];
end

% Finally, we also update the Xu parameter.
sonig.nu = sonig.nu + np;
sonig.Xu = [sonig.Xu,xu];

end