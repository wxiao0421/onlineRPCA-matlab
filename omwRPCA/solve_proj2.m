function [v, s] = solve_proj2(m, U, lambda1, lambda2)
% 
%     solve the problem:
%     min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1
%     
%     solve the projection by APG
% 
%     Parameters
%     ----------
%     m: nx1 numpy array, vector
%     U: nxp numpy array, matrix
%     lambda1, lambda2: tuning parameters
%     
%     Returns:
%     ----------
%     v: px1 numpy array, vector
%     s: nx1 numpy array, vector

    % intialization
    [n, p] = size(U);
    v = zeros(p,1);
    s = zeros(n,1);
    I = eye(p);
    converged = false;
    maxIter = Inf;
    k = 0;
    % alternatively update
%     UUt =inv(U'*U + lambda1*I)*U';
    while ~converged
        k = k+1;
        vtemp = v;
        v = (U'*U + lambda1*I)\(U'*(m-s));
%         v = UUt*(m - s);       
        stemp = s;
        s = thres(m - U*v, lambda2);
        stopc = max(norm(v - vtemp), norm(s - stemp))/n;
        if stopc < 1e-6 || k > maxIter
            converged = true;
        end
    end
end
