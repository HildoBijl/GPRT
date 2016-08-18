function R = chol3(X)

[N,~,E] = size(X);
R = zeros(N,N,E);

for i = 1:E
    R(:,:,i) = chol(X(:,:,i));
end