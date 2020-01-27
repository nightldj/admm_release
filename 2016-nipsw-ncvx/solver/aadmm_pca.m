% demo 
% use the AADMM solver
% details in An Empirical Study of ADMM for Nonconvex Problems
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_pca(D, opts)

[n, m] = size(D);
D2=D'*D;


obj = @(u, v) -v'*D2*v;
%  opt condition:  
solvh = @(v, l, t) (t*eye(m) - 2*D2) \ (t*v + l);
%  opt condition: 
solvg = @(u, l, t) norm_vec(u-l/t); %update v

fA = @(x) x;
fAt = @(x) x;
fB = @(x) -x;
fb = 0;


opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn(m, 1);
x0 = x0/norm(x0);
l0 = zeros(m, 1);

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;

end


function v = norm_vec(v)
v = v/max(norm(v(:)), 1e-20);
end
