function [sol, outs] = aradmm_SVM_unwrap(D, c, svm_C, opts)

% demo SVM primal
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  1/2 ||x||^2 + C sum(max{1-cDx, 0})
D = [D ones(length(c), 1)]; %homogeneous coordiantes
A = diag(c)*D;
[n, d] = size(A);

% h(x) = C sum(max{1-x, 0})
h = @(x) svm_C*sum(max(1-x, 0));
%proximal of sum(max{1-x, 0})
proxh = @(z, t) z + max( min(1-z, t), 0);
solvh = @(u, l, t) proxh(A*u - l/t, svm_C/t);
%h(x) = 1/2 ||x||^2
g = @(x) 0.5*x'*x;
%proxh = @(z, t) z/(1+t);
%  opt condition:  (I + tA'A) x  =  A'(l+tv)
solvg = @(av, l, t) (eye(d) + A'*A*t) \ (A'*(l-av*t));
%objective
obj = @(u, v) h(A*v) + g(v);

fA = @(x) -x;
fAt = @(x) -x;
fB = @(x) A*x;
fb = 0;

opts.obj = obj;

%% initialization
x0 = randn(d, 1);
l0 = ones(size(A, 1), 1);

%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
end