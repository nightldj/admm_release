% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_lrls(D, c, np, lam1, lam2, opts)
[n, d] = size(D);
%% minimize  lam1 |W|_* + lam2/2 ||W||^2 + 1/2 ||D(W)-c||^2
obj = @(u, v) lrls_obj(v, lam1, lam2, D, c);
% min 1/2 |DW-c|^2 + t/2 ||l/t-W+V||
%  opt condition:  
D2 = D'*D;
Dc = D'*c;
solvh = @(v, l, t) (D2+t*eye(d)) \ (Dc + v*t + l); %update u
% min lam1 |V|_* + lam2/2 ||V||_F^2 + t/2 ||l/t-W+V|| 
%  opt condition
solvg = @(u, l, t) prox_trace_norm((t*u-l)/(t+lam2), lam1/(t+lam2)); %update v


opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = randn([d, np]);
l0 = ones(size(x0));

tic;
[sol, outs] = aadmm_core(solvh, solvg, @(x) x, @(x) x, @(x)-x, 0, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;
end

function obj = lrls_obj(W, lam1, lam2, D, c)
%lam1 |W|_* + lam2/2 ||W||^2 + 1/2 ||D(W)-c||^2
[~,S,~] = svd(W,'econ');
s = diag(S);
obj = lam1*sum(s) + 0.5*lam2*norm(W, 'fro')^2 + 0.5*norm(D*W-c, 'fro')^2;
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