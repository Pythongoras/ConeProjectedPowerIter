% The power iteration function as algorithm 2.
%
% @param mat The p*p matrix to calculate eigenvector from.
% @param proj_func The projection function used after multiplication in every iteration.
%
% @return v The estimation of first eigenvector.
function v = power_iter_func(mat, proj_func)
assert(size(mat,1) == size(mat,2));

% set the stopping criteria
delta = 10^(-6);

% start with a random vector
v1 = normrnd(0,1,[size(mat,2), 1]); 
% normalize
v1 = v1 / sum(v1.^2)^0.5;
% the opposite direction
v2 = -v1;

% iterations started with v, set the initial l2 error to be inf.
v_diff = realmax;
while (v_diff > delta)
    v0 = v1;
    % multiply v1 by mat
    v1 = mat * v1;
    % projection 
    v1 = proj_func(v1);
    % normalize
    v1 = v1 / sum(v1.^2)^0.5;
    % calculate stopping criteria
    v_diff = sum((v1-v0).^2)^0.5;
end
val1 = v1' * mat * v1;

% iterations started with -v 
v_diff = realmax;
while (v_diff > delta)
    v0 = v2;
    % multiply v1 by mat
    v2 = mat * v2;
    % projection 
    v2 = proj_func(v2);
    % normalize
    v2 = v2 / sum(v2.^2)^0.5;
    % calculate stopping criteria
    v_diff = sum((v2-v0).^2)^0.5; 
end
val2 = v2' * mat * v2;

% return the one with large absolute eigenvalue
if abs(val1) > abs(val2)
    v = v1;
else
    v = v2;
end

end



%% test 

% 1). generate a small matrix to make sure the power iter runs:
% mat = cov(normrnd(0,1,[10,10]));
% v = power_iter_func(mat,@proj_ordinary);


% 2). test to see the time of power iteration is no more than double of the 
% time of eigs, and the estimation of those two are very close. That is,
% the power_iter_func is correct.
% large n = 5000, p = 10000 ---- < 2 mins
% n = 5000;
% p = 10000;
% v = mnt_cone_eigenvec_nonsparse(p);
% mat = cov(gaussian_data_mat(n,p,v));
% 
% tic;
% vec = power_iter_func(mat, @proj_ordinary);
% t = toc
% 
% tic;
% [V, D] = eigs(mat, 1);
% t = toc
% 
% sum((V-vec).^2)^0.5

