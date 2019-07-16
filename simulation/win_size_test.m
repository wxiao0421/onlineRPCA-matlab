pms.n = 4000;
pms.n_p = 250;
pms.rvec = [10 10 10];
pms.r0 = 5;
pms.rho = 0.1;
pms.nrep = 50;

pms.burnin = 200;
pms.win_size = [];
pms.track_cp_burnin = 200;
pms.n_check_cp = 20;
pms.alpha = 0.01;
pms.proportion = 0.5;
pms.n_positive = 3;
pms.min_test_size = 100;
pms.tolerance_num = 0;
pms.factor = 1;

win_range = [[100:100:500]';[1000:500:2000]';];

len = length(win_range);
all_results = cell(1,len );
omwRPCA_result_mean = zeros( len,5);
NORST_result_mean = zeros(len,5);
for win_iter = 1: length(win_range)
    pms.win_size = win_range(win_iter,:);
    if pms.burnin < pms.win_size
        pms.burnin = pms.win_size;
    end
    all_results{win_iter} = simulation4(pms);
    
    %         temp_mean = mean(table2array(all_results{idx}.omwRPCA.result_eval),1);
    %         omwRPCA_result_mean(idx,:) = [[r_iter pms.rho]  temp_mean(2:end) ];
    %         temp_mean = mean(table2array(all_results{idx}.NORST.result_eval),1);
    %         NORST_result_mean(idx,:)=[[r_iter pms.rho]  temp_mean(2:end)];
    
end
