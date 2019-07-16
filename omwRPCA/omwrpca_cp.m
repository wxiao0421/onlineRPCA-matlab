%  The loss function is
%         min_{L,S} { 1/2||M-L-S||_F^2 + lambda1||L||_* + lambda2*||S(:)||_1}
%     based on moving window.
%
%     Parameters
%     ----------
%     M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S
%         and a low-rank matrix L.
%
%     burnin : burnin sample size. We require burnin >= win_size.
%
%     win_size : length of moving window. We require win_size <= burnin.
%
%     track_cp_burnin: the first track_cp_burnin samples generated from omwrpca algorithm will exclude
%     for track change point. Because the result may be unstable.
%
%     n_check_cp: buffer size to track changepoint.
%
%     alpha: threshold value used in the hypothesis test. Hypothesis test is applied to track subspace changing.
%     We suggest use the value 0.01.
%
%     tolerance_num: offset of numbers used in hypothesis test to track change point. A larger tolerance_num gives
%     a more robust result. We restrict tolerance_num to be a non-negative integer. The default value of
%     tolerance_num is 0.
%
%     lambda1, lambda2:tuning parameters
%
%     factor: parameter factor for PCP.
%
%     Returns
%     ----------
%     Lhat : array-like, low-rank matrix.
%
%     Shat : array-like, sparse matrix.
%
%     rvec : rank of low-rank matrix.
%
%     cp: change point.
%
%     num_sparses: the number of nonzero elements of columns of sparse
%     matrix.
%
%     References
%     ----------
%
%     Rule of thumb for tuning paramters:
%     lambda1 = 1.0/np.sqrt(m);
%     lambda2 = 1.0/np.sqrt(m);
%
function [ Lhat, Shat, rvec, cp, num_sparses ] = omwrpca_cp( M, pms )
%import java.util.LinkedList % use the queue datastructure
% parameters setting
[m,n]=size(M);

if ~isfield(pms, 'lambda1') || pms.lambda1 <= 0     pms.lambda1 = 1.0/sqrt(m);      end
if ~isfield(pms, 'lambda2') || pms.lambda2 <= 0      pms.lambda2 = 1.0/sqrt(m);      end
if ~isfield(pms, 'factor') || pms.factor <= 0      pms.factor = 1;      end
if ~isfield(pms, 'tolerance_num') || pms.tolerance_num <= 0      pms.tolerance_num = 0;      end
if ~isfield(pms, 'burnin') || pms.burnin <= 0      
    if isfield(pms, 'win_size') && pms.win_size > 0
        pms.burnin = max(100, pms.win_size);      
    else
        pms.burnin = 100;
        pms.win_size = min(ceil(n/20), pms.burnin);
    end
end
if ~isfield(pms, 'track_cp_burnin') || pms.track_cp_burnin <= 0      pms.track_cp_burnin = burnin;      end
if ~isfield(pms, 'n_check_cp') || pms.n_check_cp <= 0      pms.n_check_cp = 20;      end
if ~isfield(pms, 'alpha') || pms.alpha <= 0      pms.alpha = 0.01;      end
if ~isfield(pms, 'n_positive') || pms.n_positive <= 0      pms.n_positive = 3;      end
if ~isfield(pms, 'min_test_size') || pms.min_test_size <= 0      pms.min_test_size = floor(win_size/2);      end
if ~isfield(pms, 'proportion') || pms.proportion <= 0      pms.proportion = 0.5;      end

assert(pms.burnin >= pms.win_size, 'Parameter burnin should be larger than or equal to parameter win_size.');
assert(n >= pms.burnin, 'Parameter burnin should be less than or equal to the number of columns of input matrix. n = %d, burnin = %d', n, pms.burnin);
%pms.burnin = min(pms.burnin, n);

% calculate pcp on burnin samples and find rank r
startTime = tic;
[Lhat, Shat, niter, r] = pcp(M(:, 1:pms.burnin), [], [], pms.factor);%, 10^-6, 200
% fprintf('pcp finished. niter=%d. total time is %f s\n', niter, toc(startTime));
% initialization for omwrpca
[Uhat, sigmas_hat, Vhat] = rSVD(Lhat, r, [], 5);
Vhat=Vhat';

nonneg_c = sign(Uhat(1,:));
Uhat = Uhat .* (ones(size(Uhat,1),1)*nonneg_c);
Vhat = nonneg_c'*ones(1,size(Vhat,2)).*Vhat;

U = Uhat*sqrt(sigmas_hat);%Uhat*sqrt(diag(sigmas_hat));
Vhat_win = Vhat(:, size(Vhat, 2)-pms.win_size+1:size(Vhat, 2));
A = zeros(r, r);
B = zeros(m, r);
for iter = 1: size(Vhat_win,2)
    A = A + Vhat_win(:, iter)*Vhat_win(:, iter)'; %outer(Vhat_win(:, iter), Vhat_win(:, iter));
    B = B + (M(:, pms.burnin - pms.win_size + iter) - Shat(:, pms.burnin - pms.win_size + iter))*Vhat_win(:, iter)'; % outer(M(:, burnin - win_size + iter) - Shat(:, burnin - win_size + iter), Vhat_win(:, iter));
end

% initialization for change points tracking
% dist_num_sparses: distribution of the number of nonzero elements of columns of sparse matrix
% used for tracking change point
dist_num_sparses = zeros(1,m+1);
% buffer_num: number of nonzero elements of columns of sparse matrix in the buffer used for
% tracking change point (buffer size = n_check_cp, queue structure)
buffer_num  = []; %  buffer_num  = Linkedlist();
% buffer_flag: flags of columns of sparse matrix in the buffer used for tracking change point
% (buffer size = n_check_cp, queue structure); flag=1 - potential change point; flag=0 - normal point.
buffer_flag = [];% buffer_flag = Linkedlist();
% num_sparses, cp, rvec are returned by the function
% initialize num_sparses to track the number of nonzero elements of columns of sparse matrix
num_sparses = sum(Shat ~= 0);
% initialize change points to an empty list
cp = [];

% initialize list of rank to [r]
rvec = [r];

iter = pms.burnin+1;
while iter <= n
    mi = M(:,iter);
    [vi, si] = solve_proj2(mi, U, pms.lambda1, pms.lambda2);
    Shat = [Shat reshape(si', m, 1)];
    vi_delete = Vhat_win(:,1);
    Vhat_win = [Vhat_win(:, 2:size(Vhat_win,2)) reshape(vi', r, 1)];
    A = A + vi*vi' - vi_delete*vi_delete'; %outer(vi, vi) - outer(vi_delete, vi_delete);
    B = B + (mi - si)*vi' - (M(:, iter - pms.win_size) - Shat(:, iter - pms.win_size))*vi_delete'; %outer(mi - si, vi) - outer(M(:, iter - win_size) - Shat(:, iter - win_size), vi_delete);
   
    Lhat = [Lhat reshape((U*vi)', m, 1)];    
    U = update_col(U, A, B, pms.lambda1);
    
    num_sparses = [ num_sparses sum(si ~= 0)]; %sum(reshape(si', m, 1) ~= 0)];

    if iter>= pms.burnin + pms.track_cp_burnin && iter < pms.burnin + pms.track_cp_burnin + pms.min_test_size
        num = sum(si~=0) + 1;
        dist_num_sparses(num) = dist_num_sparses(num) + 1;
    elseif iter >= pms.burnin + pms.track_cp_burnin + pms.min_test_size
        num = sum(si ~= 0) + 1;
        buffer_num = [buffer_num num]; % buffer_num.add(num);
        pvalue = 1.0*sum(dist_num_sparses(max(num - pms.tolerance_num, 0) : length(dist_num_sparses))) / sum(dist_num_sparses);
        if pvalue <= pms.alpha
            buffer_flag = [buffer_flag 1]; % buffer_flag.add(1);
        else
            buffer_flag = [buffer_flag 0]; % buffer_flag.add(0);
        end
        if length(buffer_flag) >= pms.n_check_cp
            if length(buffer_flag) == pms.n_check_cp + 1
                dist_num_sparses(buffer_num(1)) = dist_num_sparses(buffer_num(1)) + 1;
                buffer_num(1) = []; % buffer_num.pollFirst()
                buffer_flag(1) = []; % buffer_flag.pollFirst()
            end
            nabnormal = sum(buffer_flag);
            %             nabnormal = 0;
            %             tmp = buffer_flag.clone();
            %             while  tmp.size() ~=0
            %                 nabnormal= nabnormal + tmp.removeFirst();
            %             end
            % potential change identified
            if nabnormal >= pms.n_check_cp * pms.proportion
                for k = 1 : pms.n_check_cp - pms.n_positive + 1
                    % use the earliest change point if change point exists
                    if sum(buffer_flag(k: k+pms.n_positive - 1)) == pms.n_positive
                        changepoint = iter - pms.n_check_cp + k;
%                         fprintf('the num of sparse element is %d\n', buffer_num(k));
%                         fprintf('\tnorm1: %f norm2 %f \n', norm(U*vi),norm(si,1));
                        cp = [cp changepoint];
                        Lhat = Lhat(:, 1:changepoint);
                        Shat = Shat(:, 1:changepoint);
                        M_update = M(:, changepoint+1:size(M,2));
                        num_sparses = num_sparses(1:changepoint);
                        
                        % recursively call omwrpca_cp
%                         fprintf('\nthe change point is %d.\n',changepoint);
                        pms_update = pms;
                        pms_update.burnin = min(pms.burnin, size(M_update,2)-1);
                        pms_update.win_size = min(pms.win_size, size(M_update,2)-1);
                        [Lhat_update, Shat_update, rvec_update, cp_update, num_sparses_update] = omwrpca_cp(M_update, pms_update);
                        
                        % update Lhat, Shat, rvec, num_sparses, cp
                        Lhat = [Lhat  Lhat_update];
                        Shat = [Shat  Shat_update];
                        rvec = [rvec rvec_update];
                        num_sparses = [num_sparses num_sparses_update];
                        for iter_cp_update = 1 : size(cp_update)
                            cp = [cp changepoint + cp_update(iter_cp_update)];
                        end
                        return;
                    end
                end
            end
        end
    end    
    iter = iter + 1;
end
end

