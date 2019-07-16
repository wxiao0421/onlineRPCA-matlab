function [sim_result] = simulation3(pms)
omwRPCA_result.run_times = zeros(pms.nrep, 1);
omwRPCA_result.result_rank = cell(pms.nrep, 1);
omwRPCA_result.result_cp = cell(pms.nrep, 1);
omwRPCA_result.result_eval = [];%cell(pms.nrep, 1);
NORST_result.result_rank = cell(pms.nrep, 1);
NORST_result.result_eval = [];%cell(pms.nrep, 1);
NORST_result.run_times = zeros(pms.nrep, 1);
for rep_iter = 1: pms.nrep
    % simulate the data
    r_len = length(pms.rvec);
    n_piece = floor(pms.n/r_len);
    U0 = cell(r_len,1);
    for r_iter = 1: r_len
        U0{r_iter} = randn(pms.m, pms.rvec(r_iter));
    end
    V0_burnin = randn(pms.burnin, pms.rvec(1));
    V0 =  cell(r_len,1);
    for r_iter = 1: r_len
        if r_iter == r_len
            V0{r_iter} =  randn(pms.n-n_piece*(r_iter - 1), pms.rvec(r_iter));
        else
            V0{r_iter} =  randn(n_piece, pms.rvec(r_iter));
        end
    end
    L0 = U0{1}*V0_burnin';
    for r_iter = 1: r_len
        L0 = [L0 U0{r_iter}* (V0{r_iter})'];
    end
    S0 = (rand(pms.m, pms.n + pms.burnin) < pms.rho) .* (1000*rand(pms.m, pms.n + pms.burnin) - 1000);
    M0 = L0+S0;
    
    %% omwRPCA
    pms.lambda1=1.0/sqrt(pms.m);
    pms.lambda2=1.0/sqrt(pms.m)*(10^2);
    start_time = tic;
    [Lhat, Shat, rank, cp, num_sparses] = omwrpca_cp(M0, pms);
    omwRPCA_result.run_times(rep_iter) = toc(start_time);
    omwRPCA_result.result_eval = [omwRPCA_result.result_eval; evaluate(Lhat, Shat, L0, S0, rank, U0, pms.burnin, rep_iter)];
    omwRPCA_result.result_cp{rep_iter} = cp;
    omwRPCA_result.result_rank{rep_iter} = rank;
    %% NORST
    preTrain = pms.burnin;
    [L, S, niter, rank] = pcp(M0(:,1:preTrain));
    mu = mean(L, 2);
    [Utemp, Stemp, ~] = svd(1 / sqrt(preTrain) * (L - repmat(mu, 1, preTrain)));
    P_init =  Utemp(:, 1 : rank);
    ev_thresh = 1e-3;
    alpha = 250;
    K = 3;
    omega = 10;
    start_time = tic;
    %     [BG, FG, L_hat, S_hat, T_hat, t_hat, P_track_full, P_track_new] = NORST_real(M0(:, preTrain + 1 : end), P_init, mu, ev_thresh, alpha, K);
    %         [L_hat_NORST, P_hat, S_hat_NORST, T_hat, t_hat, ...
    %         P_track_full, T_calc]= Offline_NORST(M0(:,preTrain+1:size(M0,2)), ...
    %         P_init, ev_thresh, alpha, K, omega);
    [L_hat_NORST, P_hat, S_hat_NORST, T_hat, t_hat, P_track_full, t_calc] = ...
        Offline_NORST(M0(:, preTrain + 1 : end), P_init, ...
        ev_thresh, alpha, K, omega);
    NORST_result.run_times(rep_iter) = toc(start_time);
    NORST_result.result_eval = [NORST_result.result_eval; evaluate([L L_hat_NORST],[S S_hat_NORST], L0, S0, rank, U0, pms.burnin, rep_iter)];
    %     result_rank{rep_iter} = rank;
    
end

sim_result.omwRPCA =omwRPCA_result;
sim_result.NORST = NORST_result;
end