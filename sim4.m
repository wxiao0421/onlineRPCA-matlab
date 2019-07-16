% simulation C: Slowly Changing Subspace with Change Points
close all; clc; clear;
pms.m = 400;
pms.n = 3000;
pms.n_p = 250;
pms.r = [];
pms.r0 = 5;
pms.rho = [];
pms.nrep = 1;

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

r_range = [[ 10 10 10]; ];%[10,50,25];[50,50,50];
rho_range = [ 0.1; 0.01];%0.01

len = length(r_range)*size(rho_range,1);
all_results = cell(1,len );
omwRPCA_result_mean = zeros( len,5);
NORST_result_mean = zeros(len,5);
for r_iter = 1: length(r_range)
    pms.rvec = r_range(r_iter,:);
    for rho_iter = 1: length(rho_range)
        pms.rho = rho_range(rho_iter);
        idx = (r_iter - 1)*length(rho_range) + rho_iter;
        
        all_results{idx} = simulation4(pms);
        
%         temp_mean = mean(table2array(all_results{idx}.omwRPCA.result_eval),1);
%         omwRPCA_result_mean(idx,:) = [[r_iter pms.rho]  temp_mean(2:end) ];
%         temp_mean = mean(table2array(all_results{idx}.NORST.result_eval),1);
%         NORST_result_mean(idx,:)=[[r_iter pms.rho]  temp_mean(2:end)];
    end
end

