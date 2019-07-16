% Randomized SVD (Halko et al., 2009)
%     Parameters
%     ----------
%     A : Matrix to decompose
%
%     n_components: number of singular values and vectors to extract
%
%     n_oversamples: Additional number of random vectors to sample the range of M so as to ensure proper conditioning.
%         The total number of random vectors used to find the range of M is n_components + n_oversamples.
%         Smaller number can improve speed but can negatively impact the quality of approximation of singular vectors and singular values.
%
%     n_iter: Number of power iterations. It can be used to deal with very noisy problems.
%
%     power_iteration_normalizer : ¡®QR¡¯, ¡®LU¡¯
%         Whether the power iterations are normalized with step-by-step QR factorization (the slowest but most accurate),
%         or ¡®LU¡¯ factorization (numerically stable but can lose slightly in accuracy).
%         The ¡®auto¡¯ mode applies no normalization if n_iter <= 2 and switches to LU otherwise.
%
%     Returns
%     ----------
%     U : Left singular matrix.
%
%     S : Singular matrix.
%
%     V : Right singular matrix.
%
%     References
%     ----------
%     Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions Halko, et al., 2009

function [U, S, V] = rSVD(A, n_components, n_oversamples, n_iter, power_iteration_normalizer)
if ~exist('n_oversamples', 'var') || isempty(n_oversamples)
    n_oversamples = 10;
end
if ~exist('n_iter', 'var') || isempty(n_iter)
    n_iter = 4;
end
if ~exist('power_iteration_normalizer', 'var') || isempty(power_iteration_normalizer)
    power_iteration_normalizer = 'auto';
end
[~,n] = size(A);
W = randn(n, n_components+n_oversamples);
[Q,~] = qr(A*W, 0);

if strcmp(power_iteration_normalizer,'QR')
    for iter= 1: n_iter
        [Q, ~] = qr((A'*Q),0);
        [Q, ~] = qr((A*Q),0);
    end
elseif strcmp(power_iteration_normalizer,'LU') || (strcmp(power_iteration_normalizer, 'auto') && n_iter > 2)
    for iter= 1: n_iter
        [Q, ~] = lu((A'*Q));
        [Q, ~] = lu((A*Q));
    end
    %[Q, ~] = qr((A'*Q));
    [Q, ~] = qr(Q,0);
end

B = Q'*A;
[U, S, V] = svd(B, 'econ');
U = Q*U;
U = U(:, 1:n_components);
S = S(1:n_components, 1:n_components);
V = V(:, 1:n_components);
end