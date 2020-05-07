clear; close all; clc

% Set the value of p and n
p = 1000;
n = int32(10*log(p));
% Set the number of validation set used for hyperparameter tuning
vali_num = 50;

% Set the eigenvec generating function
% Simulation1: nonsparse monotone
% eigen_gen_func = @mnt_cone_eigenvec_nonsparse;

% Simulation2: sparse monotone as [0,...,0,1,...,1]
eigen_gen_func = @mnt_cone_eigenvec_sparse01;


%% Hyperparameter tuning for elasticNet SPCA

% Set up the grid of the possible hyperparameters
elas_lambda_list = linspace(0.01,10,10);
% elas_lambda_list = [];

% Initialize the container of loss for every combination of hyperparameters
elas_loss = zeros(1, length(elas_lambda_list));

for i = 1:vali_num
    
    % generate the non sparse monotone first principle eigenvector with dimension p
    eigenvec = eigen_gen_func(p);
    % generate covariance matrix
    C = cov(gaussian_data_mat(n,p,eigenvec));
    
    % alpha and lambda in elasticNet SPCA
    for j = 1:length(elas_lambda_list)
        vec = elas_spca_func(C, elas_lambda_list(j));
        elas_loss(j) = elas_loss(j) + min(sum((eigenvec + vec).^2),sum((eigenvec - vec).^2))^0.5;
    end
    
end
disp(elas_loss)

% pick out the combination of hyperparameters with minimum loss
[~,idx_lambda] = min(elas_loss);
elas_lambda_opt = elas_lambda_list(idx_lambda);
disp(elas_lambda_opt)


%% Hyperparameter tuning for truncated power iteration

% Set up the grid of the hyperparameters
trunc_cardi_list = linspace(0.05,1,20);

% Initialize the container of loss for every cardi
trunc_loss = zeros(length(trunc_cardi_list),1);

for i = 1:vali_num
    % generate the non sparse monotone first principle eigenvector with dimension p
    eigenvec = mnt_cone_eigenvec_sparse01(p);
    % generate covariance matrix
    C = cov(gaussian_data_mat(n,p,eigenvec));
    
    % cardi in truncated power iteration
    for l = 1:length(trunc_cardi_list)
        vec = power_iter_func(C,@(x) proj_trunc(x,trunc_cardi_list(l)));
        trunc_loss(l) = trunc_loss(l) + min(sum((eigenvec + vec).^2),sum((eigenvec - vec).^2))^0.5;
    end
end
disp(trunc_loss)

% pick out the combination of hyperparameters with minimum loss
[~,idx_cardi] = min(trunc_loss);
trunc_cardi_opt = trunc_cardi_list(idx_cardi);
disp(trunc_cardi_opt)


%% Experiments
% clear; close all; clc

% Set the eigenvec generating function
% Simulation1: nonsparse monotone
% eigen_gen_func = @mnt_cone_eigenvec_nonsparse;

% Simulation2: sparse monotone as [0,...,0,1,...,1]
eigen_gen_func = @mnt_cone_eigenvec_sparse01;

% Set the value of p and n
p = 1000;
n = int32(10*log(p));
% Set the number of repeat time of experiments
exp_num = 50;

% container of the run time and l2 error
run_time = zeros(1,4);
l2_error = zeros(1,4);
pev = zeros(1,4);

% The hyperparameters
trunc_cardi = 1;
elas_lambda = 3.5;

for i = 1:exp_num
    
    disp(i)
    
    % generate the non sparse monotone first principle eigenvector with dimension p
    eigenvec = eigen_gen_func(p);
    % generate covariance matrix
    data_mat = gaussian_data_mat(n,p,eigenvec);
    C = cov(data_mat);
    
    % 1) power iteration: cone proj
    t = cputime;
    v_cone = power_iter_func(C, @proj_mnt);
    run_time(1) = run_time(1) + cputime - t;
    l2_error(1) = l2_error(1) + min(sum((eigenvec-v_cone).^2), sum((eigenvec+v_cone).^2))^0.5;
    pev(1) = pev(1) + var(data_mat * v_cone);
    % 2) power iteration: ordinary
    t = cputime;
    v_ordinary = power_iter_func(C, @proj_ordinary);
    run_time(2) = run_time(2) + cputime - t;
    l2_error(2) = l2_error(2) + min(sum((eigenvec-v_ordinary).^2), sum((eigenvec+v_ordinary).^2))^0.5;
    pev(2) = pev(2) + var(data_mat * v_ordinary);
    % 3) power iteration: truncated
    t = cputime;
    v_trunc = power_iter_func(C, @(x) proj_trunc(x, trunc_cardi));
    run_time(3) = run_time(3) + cputime - t;
    l2_error(3) = l2_error(3) + min(sum((eigenvec-v_trunc).^2), sum((eigenvec+v_trunc).^2))^0.5;
    pev(3) = pev(3) + var(data_mat * v_trunc);
    % 4) elastic spca 
    t = cputime;
    v_elas = elas_spca_func(C, elas_lambda);
    run_time(4) = run_time(4) + cputime - t;
    l2_error(4) = l2_error(4) + min(sum((eigenvec-v_elas).^2), sum((eigenvec+v_elas).^2))^0.5;
    pev(4) = pev(4) + var(data_mat * v_elas);
     
end

% take the average of run time and l2 error
run_time = run_time / exp_num;
l2_error = l2_error / exp_num;
pev = pev / exp_num;

disp([n, p])
disp(run_time)
disp(l2_error)
disp(pev)





