% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_svm_dual(D, lbl, svm_C, opts)

A = diag(lbl)*D;
Q = rbf_kernel(D, D, 0.5).*(lbl*lbl');
[n, d] = size(A);
%% minimize  1/2 ||x||^2 + C sum(max{1-Ax, 0})
%dual min 1/2 a'Qa - 1'a s.t. y'a =0 0<=a<=c
% ADMM min 1/2 a'Qa - 1'a + select(b~[0,C]) s.t. y'a =0 a=b

%% problem define

fA = @(x) [x; lbl'*x];
fAt = @(x) x(1:end-1) + x(end)*lbl;
fB = @(x) - [x; 0];
fBt = @(x) - x(1:end-1);
fb = 0;
A2 = eye(n) + lbl*lbl';

 

%1/2 a'Qa - 1'a
h = @(x) 0.5*x'*Q*x - sum(x);  
% Select(b)
g = @(x) max(x-svm_C > 1e-20 | -x > 1e-20)*realmax('double');
%objective
obj = @(u, v) h(u);
%obj = @(u, v) h(u) + g(u);

solvh = @(v, l, t) (A2*t + Q)\(1 + fAt(l - t*fB(v))); 

solvg = @(u, l, t) max(0, min( fBt(l/t - fA(u)), svm_C)); %update v

opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn(n, 1);
l0 = ones(n+1, 1);

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;
end

function knl = rbf_kernel(ftr1, ftr2, sigma)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = L2_distance_2(ftr1', ftr2');
%div = 2*sigma*sigma;
%div = sigma*size(ftr1, 2);
div = sigma*median(knl(:));
knl = exp(-knl/div);
end

function n2 = L2_distance_2(x,c,df)
if nargin < 3
    df = 0;
end
[dimx, ndata] = size(x);
[dimc, ncentres] = size(c);
if dimx ~= dimc
    error('Data dimension does not match dimension of centres')
end
n2 = (ones(ncentres, 1) * sum((x.^2), 1))' + ...
    ones(ndata, 1) * sum((c.^2),1) - ...
    2.*(x'*(c));
n2 = real(full(n2));
n2(n2<0) = 0;
if (df==1)
    n2 = n2.*(1-eye(size(n2)));
end
end
