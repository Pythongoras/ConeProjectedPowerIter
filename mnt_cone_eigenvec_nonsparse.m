% The function to generate monotone non-sparse eigenvector. Equal jump with
% dimension p. Unit vector.
function v = mnt_cone_eigenvec_nonsparse(p)
v = reshape(linspace(1,p,p), p, 1);
v = v / sum(v.^2)^0.5;
end

% test 
% mnt_cone_eigenvec2(9)