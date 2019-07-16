function [output] = evaluate(Lhat, Shat, L0, S0, rank, U0, burnin, rep_iter)
% initialization
[m, n] = size(Lhat);
output = table;
output.t = (1: n - burnin)';
error_L = zeros(n - burnin, 1);
error_S = zeros(n - burnin, 1);
false_S = zeros(n - burnin, 1);
sum1_L = 0;
sum2_L = 0;
sum1_S = 0;
sum2_S = 0;
nfalse = 0;

% main loop
for iter = burnin+1 : n
    %error_L = ||Lhat-L0||_F/||L0||_F
    sum1_L = sum1_L + norm(Lhat(:,iter) - L0(:,iter));
    sum2_L = sum2_L + norm(L0(:,iter));
    error_L(iter - burnin) = (sum1_L)/(sum2_L);
    % error_S = ||Shat-S0||_F/||S0||_F
    sum1_S = sum1_S + norm(Shat(:,iter) - S0(:,iter));
    sum2_S = sum2_S + norm(S0(:,iter));
    error_S(iter - burnin) = (sum1_S)/(sum2_S);
    % number of misclassified entries in Shat
    nfalse = nfalse + sum((S0(:,iter)  ~= 0) ~= (Shat(:,iter)  ~= 0));
    false_S(iter - burnin) = nfalse;
end
output.error_L = error_L;
output.error_S = error_S;
output.false_S = false_S./(m*n);
end