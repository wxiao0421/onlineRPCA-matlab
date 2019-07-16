function [sim_result] = simulation1(pms)
omwRPCA_result.run_times = zeros(pms.nrep, 1);
omwRPCA_result.result_rank = cell(pms.nrep, 1);
omwRPCA_result.result_cp = cell(pms.nrep, 1);
omwRPCA_result.result_eval = [];%cell(pms.nrep, 1);
NORST_result.result_rank = cell(pms.nrep, 1);
NORST_result.result_eval = [];%cell(pms.nrep, 1);
NORST_result.run_times = zeros(pms.nrep, 1);
for rep_iter = 1: pms.nrep
    % simulate the data
    U0 = randn(pms.m, pms.r);
    V0 = randn(pms.n + pms.burnin, pms.r);
    L0 = U0*V0';
    S0 = (rand(pms.m, pms.n + pms.burnin) < pms.rho) .* (1000*rand(pms.m, pms.n + pms.burnin) - 1000);
    M0 = L0+S0;
    
    %% omwRPCA
%     pms.lambda1=1.0/sqrt(pms.m);
%     pms.lambda2=1.0/sqrt(pms.m)*(10^2);
%     start_time = tic;
%     [Lhat, Shat, rank, cp, num_sparses] = omwrpca_cp(M0, pms);
%     omwRPCA_result.run_times(rep_iter) = toc(start_time);
%     omwRPCA_result.result_eval = [omwRPCA_result.result_eval; evaluate(Lhat, Shat, L0, S0, rank, U0, pms.burnin, rep_iter)];
%     omwRPCA_result.result_cp{rep_iter} = cp;
%     omwRPCA_result.result_rank{rep_iter} = rank;
    %% NORST
    preTrain = pms.burnin;
    [L, S, niter, rank] = pcp(M0(:,1:preTrain));
    mu = mean(L, 2);
    [Utemp, Stemp, ~] = svd(1 / sqrt(preTrain) * (L - repmat(mu, 1, preTrain)));
    P_init =  Utemp(:, 1 : rank);
    ev_thresh = 1e-3;
    alpha = 250;
    K = 3;
    omega = 50;
    start_time = tic;
    %     [BG, FG, L_hat, S_hat, T_hat, t_hat, P_track_full, P_track_new] = NORST_real(M0(:, preTrain + 1 : end), P_init, mu, ev_thresh, alpha, K);
%         [L_hat_NORST, P_hat, S_hat_NORST, T_hat, t_hat, ...
%         P_track_full, T_calc]= Offline_NORST(M0(:,preTrain+1:size(M0,2)), ...
%         P_init, ev_thresh, alpha, K, omega);
    [L_hat_NORST, P_hat, S_hat_NORST, T_hat, t_hat, P_track_full, t_calc] = ...
        NORST(M0(:, preTrain + 1 : end), P_init, ...
        ev_thresh, alpha, K, omega);
    NORST_result.run_times(rep_iter) = toc(start_time);
    NORST_result.result_eval = [NORST_result.result_eval; evaluate([L L_hat_NORST],[S S_hat_NORST], L0, S0, rank, U0, pms.burnin, rep_iter)];
    %     result_rank{rep_iter} = rank;
    
end

sim_result.omwRPCA =omwRPCA_result;
sim_result.NORST = NORST_result;
end