function B = solve_cholE(R, A, tr)
% Wrapper function to call solve_chol for matrices of size
%   R: I-by-I-by-J
%   A: I-by-K-by-L-by-M, where L = J or L = 1.
% output is then:
%   B: I-by-K-by-J-by-M

[I K L M] = size(A); J = size(R,3); B = zeros(I, K, J, M);
if L ~=1 && L ~= J; error('3rd dimension of A must be singleton or match R'); end

if nargin == 2
    for i=1:J
        for j = 1:M
            B(:,:,i,j) = solve_chol(R(:,:,min(J,i)),A(:,:,min(L,i),j));
        end
    end
        
else
    % solve for A*inv(X)
    for i=1:J
        for j = 1:M
            B(:,:,i,j) = solve_chol(R(:,:,i),A(:,:,min(a(3),i),min(a(4),j))')';
        end
    end
end
    