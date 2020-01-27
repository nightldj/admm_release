% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_qp(Q, q, A, b, opts)
[nb, nu] = size(A);
%% minimize  1/2 u'Qu+q'u s.t. Au<=b
% H(u) = 1/2 u'Qu+q'u, G(v) = Select(v<=b)
%1/2 u'Qu+q'u
h = @(x) 0.5*x'*Q*x+q'*x;  
% Select(v<=b)
g = @(x) max(x-b > 1e-20)*realmax('double');
%objective
%obj = @(u, v) h(u);
obj = @(u, v) h(u)+g(A*u);
%  opt condition:  (tA'A+Q) x  =  A'(l+tv)-q
solvh = @(v, l, t) (A'*A*t+Q)\(A'*(l+t*v)-q); 
%  opt condition:  min(Au-l/t, b)
solvg = @(u, l, t) min(A*u-l/t, b); %update v
fA = @(x) A*x;
fAt = @(x) A'*x;
fB = @(x) - x;
fb = 0;


opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn(nb,1);
l0 = ones(size(x0));

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;
end
