function [sol, outs] = aradmm_elasticnet(M, f, lam1, lam2, opts)

% demo linear regression with elastic net regularizer
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  lam1 |x| + lam2/2 ||x||^2 + 1/2 ||Mx-f||^2

%% subproblem solver
[n, m] = size(M);
%1/2 |Mx-f|^2
h = @(x) norm(M*x-f)^2/2;  %regression term
%lam1 |x| + lam2/2 |x|^2
g = @(x) lam1*sum(abs(x)) + x'*x*lam2/2; %regularizer
obj = @(u, v) h(v)+g(v);
% min 1/2 |Mx-f|^2 + 1/(2t)|| x - z ||^2
%  opt condition:  (M'M+I/t) x  =   z/t + M'f
Mf = M'*f;
if n<m % M is fat
    M2 = M*M';
    proxh = @(z, t) (z+t*Mf) - t*M'*((t*M2+eye(n))\(M*(z+t*Mf)));  %Woodbury Matrix Identity
else
    M2 = M'*M;
    proxh = @(z,t) (t*M2+eye(m))\(z+t*Mf);
end
solvh = @(v, l, t) proxh(v+l/t, 1/t); %update u
% min lam1 |x| + lam2/2 |x|^2 + 1/(2t)|| x - z ||^2
%  opt condition:  shrink(z, lam1*t)/(1+lam2*t)
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
proxg = @(z,t) shrink(z, lam1*t)/(1+lam2*t);
solvg = @(u, l, t) proxg(u-l/t, 1/t); %update v
fA = @(x) x;
fAt = @(x) x;
fB = @(x) -x;
fb = 0;

opts.obj = obj; %objective function, used when verbose


%% initialization
x0 = randn(m,1);
l0 = ones(size(x0));

%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;

