% The projection function of truncated power iteration. To retain the
% largest cardi*p coordinates.
%
% @param v0 The vector to be projected.
% @param cardi The proportion of non-zero coordinates.
function v = proj_trunc(v0, cardi)
% the actual number of non-zero coordinates
cutoff_idx = int32(cardi*length(v0));
% copy the input v0 to v
v = v0;
% sort v0 in descend order, and get the smallest value of the retained
% coordinates as the cutoff value
v0 = sort(abs(v0), 'descend');
cutoff = v0(cutoff_idx);
% for every coordinate of v, if the value < cutoff, set it to zero
v(find(abs(v)<cutoff)) = 0;
end