% simulation B: slowly changing subspace
close all; clc; clear;
pms.m = 400;
pms.n = 5000;
pms.n_p = 250;
pms.r0 = 5;
pms.r = [];
pms.rho = [];
pms.nrep =1;

pms.burnin = 200;
pms.win_size = 200;
pms.track_cp_burnin = 200;
pms.n_check_cp = 20;
pms.alpha = 0.01;
pms.proportion = 0.5;
pms.n_positive = 3;
pms.min_test_size = 100;
pms.tolerance_num = 0;
pms.factor = 1;

r_range = [10 50];
rho_range = [ 0.01 0.1];

len = length(r_range)*length(rho_range);
all_results = cell(1,len );
omwRPCA_result_mean = zeros( len,5);
NORST_result_mean = zeros(len,5);
for r_iter = 1: length(r_range)
    pms.r = r_range(r_iter);
    for rho_iter = 1: length(rho_range)
        pms.rho = rho_range(rho_iter);
        idx = (r_iter - 1)*length(rho_range) + rho_iter;        

        [all_results{idx}] = simulation2(pms);
        
%         temp_mean = mean(table2array(all_results{idx}.omwRPCA.result_eval),1);
%         omwRPCA_result_mean(idx,:) = [[pms.r pms.rho]  temp_mean(2:end) ];
%         temp_mean = mean(table2array(all_results{idx}.NORST.result_eval),1);
%         NORST_result_mean(idx,:)=[[pms.r pms.rho]  temp_mean(2:end)];
    end
end

