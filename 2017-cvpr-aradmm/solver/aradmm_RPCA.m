function [sol, outs] = aradmm_RPCA(C, lam1, opts)

% demo Robust PCA
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com


%% minimize_{A, E} |A|_* + lam1 |E| st A+E=C
obj = @(A, E) rpca_obj(A, E, lam1);

solvh = @(v, l, t) prox_trace_norm(C-v+l/t, 1/t); %update u
% min lam1 |V|_* + lam2/2 ||V||_F^2 + t/2 ||l/t-W+V|| 
%  opt condition
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
solvg = @(u, l, t) shrink(C-u+l/t, lam1/t); %update v
fA = @(x) x;
fAt = @(x) x;
fB = @(x) x;
fb = C;

opts.obj = obj;

%% initialization
x0 = randn(size(C));
l0 = ones(size(x0));


%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
end

function obj = rpca_obj(A, E, lam1)
[~,S,~] = svd(A,'econ');
s = diag(S);
obj = lam1 * norm(E(:), 1) + sum(abs(s)) ;
end

function oA = prox_trace_norm(A, t)
% min ||U||_* + 0.5/t ||U-A||_F
[U,S,V] = svd(A,'econ');
s = diag(S);
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
os = shrink(s, t);
oS = diag(os);
oA = U*oS*V';
end