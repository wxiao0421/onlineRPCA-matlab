% Online Moving Window Robust PCA
%
%     The loss function is
%         min_{L,S} { ||L||_* + lam*||S(:)||_1 + 1/{2*mu}||M-L-S||_F^2}
%
%     Parameters
%     ----------
%     M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S
%         and a low-rank matrix L.
%
%     burnin : burnin sample size.
%
%     win_size : length of moving window. We require win_size <= burnin.
%
%     lambda1, lambda2:tuning parameters of online moving windows robust PCA.
%
%     mu: postive tuning parameter (default NaN). When mu is set to NaN, the value sqrt(2*max(m, n))
%     will be used in the algorithm, where M is a m by n matrix. A good choice of mu is sqrt(2*max(m, n))*sigma,
%     where sigma is the standard deviation of error term.
%
%     Returns
%     ----------
%     Lhat : array-like, low-rank matrix.
%
%     Shat : array-like, sparse matrix.
%
%     rank : rank of low-rank matrix.
%
%     References
%     ----------
%
%     Rule of thumb for tuning paramters:
%     ----------
%     lambda1 = 1.0/np.sqrt(m);
%     lambda2 = 1.0/np.sqrt(m);
%

function [Lhat, Shat, r] = omwrpca(M, pms)

[m,n]=size(M);

if ~isfield(pms, 'lambda1') || pms.lambda1 <= 0     pms.lambda1 = 1.0/sqrt(m);      end
if ~isfield(pms, 'lambda2') || pms.lambda2 <= 0      pms.lambda1 = 1.0/sqrt(m);      end
if ~isfield(pms, 'factor') || pms.factor <= 0      pms.factor = 1;      end
if ~isfield(pms, 'burnin') || pms.burnin <= 0      
    if isfield(pms, 'win_size') && pms.win_size > 0
        pms.burnin = max(100, pms.win_size);      
    else
        pms.burnin = 100;
        pms.win_size = min(ceil(n/20), pms.burnin);
    end
end

assert(pms.burnin >= pms.win_size, 'Parameter burnin should be larger than or equal to parameter win_size.');
assert(n >= pms.burnin, 'Parameter burnin should be less than or equal to the number of columns of input matrix.');

% calculate pcp on burnin samples and find rank r
startTime = tic;
[Lhat, Shat, niter, r] = pcp(M(:, 1:pms.burnin), [], [], pms.factor, 10^-4, 150);
fprintf('pcp finished. niter=%d. total time is %f s\n', niter, toc(startTime));
% figure;
% temp=reshape(Lhat(:,1),51,[]);
% imshow(temp);
% initialization for omwrpca
[Uhat, sigmas_hat, Vhat] = rSVD(Lhat, r, [], 5);
Vhat=Vhat';
U = Uhat*sqrt(sigmas_hat);%Uhat*sqrt(diag(sigmas_hat));
Vhat_win = Vhat(:, size(Vhat, 2)-pms.win_size+1:size(Vhat, 2));
A = zeros(r, r);
B = zeros(m, r);
for iter = 1: size(Vhat_win,2)
    A = A + Vhat_win(:, iter)*Vhat_win(:, iter)'; %outer(Vhat_win(:, iter), Vhat_win(:, iter));
    B = B + M(:, pms.burnin - pms.win_size + iter) - Shat(:, pms.burnin - pms.win_size + iter)*Vhat_win(:, iter)'; % outer(M(:, burnin - win_size + iter) - Shat(:, burnin - win_size + iter), Vhat_win(:, iter));
end

% main loop
for iter = pms.burnin+1 : n
    mi = M(:, iter);
    [vi, si] = solve_proj2(mi, U, pms.lambda1, pms.lambda2);
    Shat = [Shat  reshape(si,m,1)];
    vi_delete = Vhat_win(:,1);
    Vhat_win = [Vhat_win(:,1:size(Vhat_win,2)) reshape(vi,r,1)];
    A = A + vi*vi' - vi_delete*vi_delete'; %outer(vi, vi) - outer(vi_delete, vi_delete);
    B = B + (mi - si)*vi' - (M(:, iter - pms.win_size) - Shat(:, iter - pms.win_size))*vi_delete'; %outer(mi - si, vi) - outer(M(:, iter - win_size) - Shat(:, iter - win_size), vi_delete);
    
    Lhat = [Lhat reshape(U*vi,m,1)];
    U = update_col(U, A, B, pms.lambda1);
end
end
