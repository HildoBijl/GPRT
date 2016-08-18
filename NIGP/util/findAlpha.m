function alpha = findAlpha(R,y)

alpha = zeros(size(y));

for i=1:size(y,2);
    alpha(:,i) = solve_chol(R(:,:,i),y(:,i));
end