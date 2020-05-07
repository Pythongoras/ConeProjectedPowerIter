% The function to generate an n*p gaussian matrix with covariance matrix = I + kvv^T
function mat = gaussian_data_mat(n,p,v)
% assert(length(v) == p)

% the eigengap k == log(p)
k = log(p);
% sqrt{I + kvv^T} = I + (sqrt{k+1}-1)vv^T
C_sqrt = eye(p) + ((k+1).^0.5-1) * (v*v');
% generate the gaussian matrix
mat = normrnd(0,1,[n,p]) * C_sqrt;
end