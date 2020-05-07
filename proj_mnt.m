% The projection function of the Monotone Cone Projected Power Iteration.
function v = proj_mnt(v0)
p = length(v0);
% x can be any monotonely increasing vector. Here we generate x as (1,2,..,p)
x = linspace(1,p,p)';
% perform isotonic regression on v0 to get its projection into monotone
% cone
v = lsqisotonic( x,v0,ones(1,p) );
end