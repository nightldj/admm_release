% demo basis pursuit
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_basis_pursuit(A, b, opts)

%% subproblem solver
[m, n] = size(A);
% Select(Au=b)
h = @(x) max(norm(A*x-b) > sqrt(length(b))*1e-20)*realmax('double');
% |v|
g = @(x) sum(abs(x));
%objective
obj = @(u, v) g(v);
%  opt condition:  x = (I - A' inv(AA') A)(v + l/t) + A' inv(AA') b
AAt = A*A';
P = eye(n) - A' * (AAt \ A);
q = A' * (AAt \ b);
solvh = @(v, l, t) P*(v + l/t) + q;
%  opt condition:
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
solvg = @(u, l, t) shrink(u - l/t, 1/t);
% constraint
fA = @(x) x;
fAt = @(x) x;
fB = @(x) - x;
fb = 0;

opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn(n,1);
l0 = ones(size(x0));

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
end