function [tr] = mtrace(A, dim)
%mtrace Trace of a multi-dimensional array along specified dimensions.
% 
% Input:
% A: The matrix to take the trace of.
% dim: The dimensions along which to take the trace. (Default: [1,2])
% Suppose that dim = [2,4] for a five-dimensional matrix. Then this function returns a three-dimensional matrix B whose elements B(i,j,k) equal "A(i,1,j,1,k) + A(i,2,j,2,k) + A(i,3,j,3,k) + ... ". 

% If no inputs are provided, we show the help.
if nargin == 0
    help mtrace;
    return;
end

% If the dim parameter has not been given, we set it to its default value.
if (nargin < 2)
    dim = [1, 2];
end

% If the dim parameter is of the wrong size, we throw an error.
if numel(dim)~=2
    error('sw:sw_matmult:WrongInput','dim has to be a two element array!');
end
if dim(1) == dim(2)
    error('sw:sw_matmult:WrongInput','dim cannot target the same dimension twice!');
end
% We want dim(1) to be smaller than dim(2). If this is not the case, we switch them to make it so.
if dim(1) > dim(2)
	temp = dim(2);
	dim(2) = dim(1);
	dim(1) = temp;
end

% We look at the sizes of the matrices we've been given.
nDA = ndims(A);
nD = max([nDA dim]);

% First we permute the matrix to put the two dimensions we put the trace over at the end.
perm = [1:dim(1)-1,dim(1)+1:dim(2)-1,dim(2)+1:nD,dim(1),dim(2)];
A = permute(A, perm);
nA = [size(A),ones(1,nD-nDA)];
nSum = min(nA([end-1,end])); % This is the amount of terms we need to sum over.

% We now walk through the dimensions and set up the indices array.
ind = 1;
multip = 1;
for i = 1:nD-2
	perm = [2:i,1];
	if i == 1 % This is a fix because Matlab cannot handle one-dimensional arrays.
		perm = [1,2];
	end
	ind = bsxfun(@plus, ind, permute((0:(nA(i)-1))'*multip, perm));
	multip = multip*nA(i);
end
multip = multip*(nA(nD-1)+1);
ind = bsxfun(@plus, ind, permute((0:(nSum-1))'*multip, [2:nD-1, 1]));

% We now take all the necessary elements from A and apply the summation over the right elements.
tr = sum(A(ind), nD-1);

end

