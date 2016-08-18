function C = dmmat(A,B,dim1,dim2)
% A double matrix multiplication of multidimensional arrays along specified dimensions.
%
% Input:
% A, B      Multidimensional input arrays.
% dim1       Contains two numbers, to select two dimensions. (Default: [1 2])
% dim2       Contains two numbers, to select two dimensions. (Default: [3 4])
%
% The multiplication is a standard matrix multiplication. The two matrices are selected by dim:
%   AB = A(... dim(1) ... dim(2) ...) * B(... dim(1) ... dim(2) ...)
% The necessary condition that the multiplication can be performed:
%   size(A,dim1(2)) = size(B,dim1(1))
%   size(A,dim2(2)) = size(B,dim2(1))
%   size(A,d) = size(B,d) for d not equal to dim1(1), dim1(2), dim2(1) or dim2(2)
%
% Singleton dimensions in either A or B matrices are, for safety reasons, not supported. The sizes have to match.
%

% If no inputs are provided, we show the help.
if nargin == 0
    help mmat;
    return;
end

% If the dim parameter has not been given, we set it to its default value.
if (nargin < 3)
    dim1 = [1 2];
end
if (nargin < 4)
    dim2 = [3 4];
end

% If the dim parameter is of the wrong size, we throw an error.
if numel(dim1)~=2
    error('sw:sw_matmult:WrongInput','dim1 has to be a two element array!');
end
if numel(dim2)~=2
    error('sw:sw_matmult:WrongInput','dim2 has to be a two element array!');
end

% We look at the sizes of the matrices we've been given.
nDA = ndims(A);
nDB = ndims(B);
nD = max([nDA nDB dim1 dim2]);

% We also check if the matrix sizes match up. 
if size(A,dim1(2)) ~= size(B,dim1(1))
    error('sw:sw_matmult:WrongInput','Wrong input matrix sizes! The size of A at dim1(2) is not the size of B at dim1(1).');
end
if size(A,dim2(2)) ~= size(B,dim2(1))
    error('sw:sw_matmult:WrongInput','Wrong input matrix sizes! The size of A at dim2(2) is not the size of B at dim2(1).');
end
for i = 1:nD
	if i ~= dim1(1) && i ~= dim1(2) && i ~= dim2(1) && i ~= dim2(2) && size(A,i) ~= size(B,i)
 	  error(['sw:sw_matmult:WrongInput','Wrong input matrix sizes! The input matrices do not have the same size in dimension ',num2str(i),'.']);
	end
end

% We look at the sizes of the matrix at the dimensions at which we multiply. Note that nA(dim(2)) = nB(dim(1)) for each dim array.
nA = [size(A),ones(1,nD-nDA)];
nB = [size(B),ones(1,nD-nDB)];

% We want to multiply the dimension dim(2) of A. So what we do is we put this dimension at the end of our matrix. This creates a singleton dimension at dim(2). We will eventually multiply this
% with B(dim(2)), so we repeat the matrix until it is of the same size as B(dim(2)).
idx = 1:nD+2;
idx([dim1(2), dim2(2), end-1, end]) = idx([end-1, end, dim1(2) dim2(2)]);
A = permute(A,idx);
rep = ones(1,nD+2);
rep(dim1(2)) = nB(dim1(2));
rep(dim2(2)) = nB(dim2(2));
A = repmat(A,rep);
% We want to multiply the dimension dim(1) of B. So what we do is we put this dimension at the end of our matrix. This creates a singleton dimension at dim(1). We will eventually multiply this
% with A(dim(1)), so we repeat the matrix until it is of the same size as A(dim(1)).
idx = 1:nD+2;
idx([dim1(1), dim2(1), end-1, end]) = idx([end-1, end, dim1(1) dim2(1)]);
B = permute(B,idx);
rep = ones(1,nD+2);
rep(dim1(1)) = nA(dim1(1));
rep(dim2(1)) = nA(dim2(1));
B = repmat(B,rep);

% Now we do the multiplication. We do this element-wise. We then sum up the results over the dimension at the end, where we lined up both matrices.
C = sum(sum(A.*B, nD+2), nD+1);

end