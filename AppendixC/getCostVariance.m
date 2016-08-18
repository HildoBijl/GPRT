function [VJ] = getCostVariance(A,B,F,V,mu0,Psi0,a,Q,R)
%getCostVariance This function returns the variance of the cost for a given system. The inputs should equal (A,B,F,V,mu0,Psi0,a,Q,R).

% We calculate all the matices required for the variance, which we subsequently also find.
I = eye(size(A));
Ac = A - B*F;
if (max(real(eig(Ac+a*I))) >= 0)
	VJ = inf;
else
	Qc = Q + F'*R*F;
	XbaQ = lyap((Ac+a*I)',Qc);
	X2aPsi0 = lyap((Ac+2*a*I),Psi0);
	X2aV = lyap((Ac+2*a*I),V);
	VJ = 2*trace((Psi0*XbaQ)^2) - 2*(mu0'*XbaQ*mu0)^2 + 4*trace((X2aPsi0 - X2aV/(4*a))*XbaQ*V*XbaQ);
end

end

