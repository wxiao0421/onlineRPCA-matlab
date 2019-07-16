% Robust PCA (Candes et al., 2009)
%     
%     This code solves the Principal Component Pursuit
%     min_M { ||L||_* + lam*||S(:)||_1 }
%     s.t. M = S+L
%     using an Augmented Lagrange Multiplier (ALM) algorithm.
%   
%     Parameters
%     ----------
%     M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
%         and a low-rank matrix L.
%         
%     lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
%     will be used in the PCP algorithm, where M is a m by n matrix.
%     
%     mu: postive tuning parameter used in augmented Lagrangian which is in front of 
%     the ||M-L-S||_F^2 term (default NaN). When mu is set to NaN, the value 0.25/np.abs(M).mean() 
%     will be used in the PCP algorithm.
%     
%     factor: tuning parameter (default 1). When lam is set to NaN,  lam will take the value 1/sqrt(max(m, n)) * factor
%     in the PCP algorithm, where M is a m by n matrix. When lam is not NaN, factor is ignored.
%     
%     tol : tolerance value for convergency (default 10^-7).
%     
%     maxit : maximum iteration (default 1000).
%     
%     Returns
%     ----------
%     L : array-like, low-rank matrix.
%     
%     S : array-like, sparse matrix.
%     
%     niter : number of iteration.
%     
%     rank : rank of low-rank matrix.
%     
%     References
%     ----------
%     Candes, Emmanuel J., et al. Robust principal component analysis. 
%         Journal of the ACM (JACM) 58.3 (2011): 11.
%     
%     Yuan, Xiaoming, and Junfeng Yang. Sparse and low-rank matrix decomposition via alternating direction methods. 
%         preprint 12 (2009). [tuning method]

function [L, S, niter, rank] = pcp(M, lam, mu, factor, tol, maxit)

% initialization
[m, n] = size(M);
unobserved = isnan(M);
M(unobserved) = 0;
S = zeros(m,n);
L = zeros(m,n);
Lambda = zeros(m,n); % the dual variable
% M(1:5,1:5)
% parameters setting
if ~exist('factor', 'var') || isempty(factor)
    factor = 1;
end
if ~exist('lam', 'var') || isempty(lam)
    lam = 1.0/sqrt(max(m,n))*factor;
end
if ~exist('mu', 'var') || isempty(mu)
    mu = 0.25/mean(mean(abs(M)));
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 10^-7;
end
if ~exist('maxit', 'var') || isempty(maxit)
    maxit = 1000;
end

% main
for niter = 1 : maxit
    normLS = norm([S L], 'fro');
    % dS, dL record the change of S and L, only used for stopping criterion
    
    X = Lambda / mu + M;
    
    %  L - subproblem
    % L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
    % L has closed form solution (singular value thresholding)
    Y = X - S;
    dL = L;
    [U, sigmas, V] = svd(Y,'econ');
    rank = sum(sum(sigmas > 1/mu));
    Sigma = diag(diag(sigmas(1:rank, 1:rank)) - 1/mu);%diag(sigmas(1:rank, 1:rank) - 1/mu);
    V=V';
    L = U(:,1:rank)*Sigma*V(1:rank,:);%dot(dot(U(:,1:rank), Sigma), V(1:rank,:));
    dL = L - dL;
    
    %  S - subproblem
    %  S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
    %  Define element wise softshinkage operator as
    %       softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
    %  S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
    Y = X - L;
    dS = S;
    S = thres(Y, lam/mu); % softshinkage operator
    dS = S - dS;
    
    % Update Lambda (dual variable)
    Z = M - S - L;
    Z(unobserved) = 0;
    Lambda = Lambda + mu * Z;
    
    %  stopping criterion
    RelChg = norm([dS  dL], 'fro') / (normLS + 1);
    if RelChg < tol
        break;
    end
%     if mod(niter,10) == 1
%         fprintf('rank %d, norm %f \n', rank, normLS);
%         Sigma
%         X(1:5,1:5)
%     end
end
end
