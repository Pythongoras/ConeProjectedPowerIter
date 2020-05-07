% The elasticNet spca algorithm to compute only the first eigenvector
% in sparse version. Use the closed form in paper to solve B.
%
% @param mat The matrix to calculate the first eigenvector from.
% @param lambda The hyperparameter that controls sparsity.
%
% @return v The sparse first eigenvector of matrix mat.
function v = elas_spca_func(mat, lambda)

% the stop criterion. If the l2 difference of B_t and B_t-1 is less than
% this value, we stop the iteration.
cvgCriterion = 1e-6;
% set up the parameters
% dimension of matrix
[~, p] = size(mat);
% initialize the container of sparse eigenvector estimations B_t and B_t-1
B = zeros(p, 1);
B_prev = ones(p, 1);
% get the gram matrix to save computing time
Gram = mat.'*mat;

% get the first principal eigenvector of mat
[A, ~] = eigs(mat, 1);

% While ||B_t - B_t-1|| > cvgCriterion, do:
while sum((B - B_prev).^2)^0.5 > cvgCriterion
    
    % move the value of B_prev to B
    B_prev = B;
    
    % compute B by the closed form
    AXX = A'*Gram;
    B = ( sign(AXX).*max(0,abs(AXX)-0.5*lambda) )'; 
    
    % update A
    [U,~,V] = svd(Gram*B, 'econ');
    A = U*V';
    
    disp(sum((B - B_prev).^2)^0.5)
end

% normalize B
v = B / sum(B.^2)^0.5;
    
end




%% test

% 1) generate a small matrix to make sure the algorithm converges:
% clear; close all; clc;
% n = 10;
% p = 10;
% mat = cov(normrnd(0,1,[n,p]));
% vec = elas_spca_func(mat, 0.1);
% [V, D] = eigs(mat, 1);
% sum((V-vec).^2)^0.5

% 2). large n = 3000, p = 10000 ---- < 2 mins
% clear; close all; clc;
% n = 1000;
% p = 5000;
% v = mnt_cone_eigenvec_nonsparse(p);
% mat = cov(gaussian_data_mat(n,p,v));
% 
% tic;
% vec = elas_spca_func(mat, 0.1);
% t = toc

% tic;
% [V, D] = eigs(mat, 1);
% t = toc
% 
% sum((V-vec).^2)^0.5
% sum(vec~=0)


