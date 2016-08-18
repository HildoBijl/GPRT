function C = mmat(A,B,dim)
% Simple matrix multiplication of multidimensional arrays along specified dimensions.
%
% Input:
% A, B      Multidimensional input arrays.
% dim       Contains two numbers, to select two dimensions. (Default: [1 2])
%
% The multiplication is a standard matrix multiplication. The two matrices are selected by dim:
%   AB = A(... dim(1) ... dim(2) ...) * B(... dim(1) ... dim(2) ...)
% The necessary condition that the multiplication can be performed:
%   size(A,dim(2)) = size(B,dim(1))
%   size(A,d) = size(B,d) for d not equal to dim(1) or dim(2)
%
% Singleton dimensions in either A or B matrices are, for safety reasons, not supported. The sizes have to match.
%
% Examples:
% For 2D matrices mmat is identical to the Matlab built-in multiplication:
% A = [1 2 3 4];
% B = [1;2;3;4];
% C = mmat(A,B)
%
% C will be 30.
%
% For multidimensional arrays:
% A = repmat([1 2 3 4],[1 1 5]);
% B = repmat([1 2 3 4]',[1,1,5]);
% C = mmat(A,B)
% C will be an array with dimensions of 1x1x5 and every element is 30.
%

% If no inputs are provided, we show the help.
if nargin == 0
    help mmat;
    return;
end

% If the dim parameter has not been given, we set it to its default value.
if (nargin < 3)
    dim = [1 2];
end

% If the dim parameter is of the wrong size, we throw an error.
if numel(dim)~=2
    error('sw:sw_matmult:WrongInput','dim has to be a two element array!');
end

% We look at the sizes of the matrices we've been given.
nDA = ndims(A);
nDB = ndims(B);
nD = max([nDA nDB dim]);

% We also check if the matrix sizes match up. 
if size(A,dim(2)) ~= size(B,dim(1))
    error('sw:sw_matmult:WrongInput','Wrong input matrix sizes! The size of A at dim(2) is not the size of B at dim(1).');
end
for i = 1:nD
	if i ~= dim(1) && i ~= dim(2) && size(A,i) ~= size(B,i)
 	  error(['sw:sw_matmult:WrongInput','Wrong input matrix sizes! The input matrices do not have the same size in dimension ',num2str(i),'.']);
	end
end

% We look at the sizes of the matrix at the dimensions at which we multiply. Note that nA(dim(2)) = nB(dim(1)).
nA = [size(A),ones(1,nD-nDA)];
nB = [size(B),ones(1,nD-nDB)];

% We want to multiply the dimension dim(2) of A. So what we do is we put this dimension at the end of our matrix. This creates a singleton dimension at dim(2). We will eventually multiply this
% with B(dim(2)), so we repeat the matrix until it is of the same size as B(dim(2)).
idx = 1:nD+1;
idx([dim(2) end]) = idx([end dim(2)]);
A = permute(A,idx);
rep = ones(1,nD+1);
rep(dim(2)) = nB(dim(2));
A = repmat(A,rep);
% We want to multiply the dimension dim(1) of B. So what we do is we put this dimension at the end of our matrix. This creates a singleton dimension at dim(1). We will eventually multiply this
% with A(dim(1)), so we repeat the matrix until it is of the same size as A(dim(1)).
idx = 1:nD+1;
idx([dim(1) end]) = idx([end dim(1)]);
B = permute(B,idx);
rep = ones(1,nD+1);
rep(dim(1)) = nA(dim(1));
B = repmat(B,rep);

% Now we do the multiplication. We do this element-wise. We then sum up the results over the dimension at the end, where we lined up both matrices.
C = sum(A.*B, nD+1);

end