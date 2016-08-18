function ipK = dipK2ipK(dipK)

[N E] = size(dipK);

ipK = zeros(N,N,E);

for i=1:E
    ipK(:,:,i) = diag(dipK(:,i));
end