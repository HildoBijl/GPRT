function B = solve_cholE(R, A, tr)

[a1 a2 a3 a4] = size(A); r = size(R,3);
B = zeros(size(A));

if nargin == 2
    for i=1:size(R,3)
        for j = 1:size(A,4)
            B(:,:,i,j) = solve_chol(R(:,:,min(r,i)),A(:,:,min(a3,i),min(a4,j)));
        end
    end
        
else
    % solve for A*inv(X)
    for i=1:size(R,3)
        for j = 1:size(A,4)
            B(:,:,i,j) = solve_chol(R(:,:,i),A(:,:,min(a(3),i),min(a(4),j))')';
        end
    end
end
    