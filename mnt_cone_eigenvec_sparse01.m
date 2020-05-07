function v = mnt_cone_eigenvec1(p)
cardi = 0.3;
v = [zeros(p-int32(p*cardi),1); ones(int32(p*cardi),1)];
v = v / sum(v.^2)^0.5;
end


% test 
% mnt_cone_eigenvec1(9)