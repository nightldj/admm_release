% demo 
% use the AADMM solver
% details in An Empirical Study of ADMM for Nonconvex Problems
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_phase(D, c, opts)

[n, m] = size(D);
%%max_l min 0.5*||abs(u)-c||^2 +  Re<Dv-u, l> +  t/2  ||Dv-u||^2,
obj = @(u, v) 0.5*norm(abs(D*v)-c)^2;
%  opt condition:  
solvh = @(v, l, t) solv_complex(D,c,v,l,t);
%  opt condition: 
%DtD = D'*D;
%solvg = @(u, l, t) D\(u-l/t);
Dinv = pinv(D);
solvg = @(u, l, t) Dinv*(u-l/t);

fA = @(x) x;
fAt = @(x) x;
fB = @(x) -D*x;
fb = 0;


opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn(m, 1)+1i*randn(m, 1);
l0 = ones(n, 1);

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;

end


function u = solv_complex(D,c,v,l,t)
%% min_u 0.5*||abs(u)-c||^2 + t/2  ||Dv-u + l/t||^2

s = D*v+l/t;
u = t*(abs(s))+c;
u = u/(t+1);
u = u.*sign(s);

end