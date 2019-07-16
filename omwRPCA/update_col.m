function [U] = update_col(U, A, B, lambda1)
[m, r] = size(U);
A = A + lambda1*eye(r);
for j = 1 : r
    bj = B(:,j);
    uj = U(:,j);
    aj = A(:,j);
    temp = (bj - U*aj)/A(j,j) + uj;
    U(:,j) = temp/max(norm(temp), 1);
end

end