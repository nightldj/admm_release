% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_sdp_dual(C, A, b, opts)

%% minimize  <C, X> st A(X) = b, X>=0
%dual min -b'y st A^*(y) + S = C, S >=0
%y -> u, S -> v, X -> l
% min -b'u st A*u + v = C, v>=0
[n, ~] = size(C);
[m, ~] = size(A);
%% problem define

obj = @(u, v) -b'*v;
%  opt condition:  
solvh = @(u, l, t) sd_project(C - reshape(A'*u, n, n) + l/t); %update u
%  opt condition:  tAA' y  =   Ax + b - t A(S-C)
solvg = @(v, l, t) (A*A') \ ((A*l(:) + b)/t - A*(v(:)-C(:))); %update v

fA = @(x) x;
fAt = @(x) x;
fB = @(x) reshape(A'*x, n, n);
fb = C;

opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = zeros(m, 1);
l0 = zeros(n);

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.v;
end

function oA = sd_project(A)
% positive semi-definite projection
A = A*0.5 + A'*0.5;   %%to guarantee symmetric?
[V, D] = eig(A);
s = real(diag(D));
s(s<0) = 0;
oS = diag(s);
V = real(V);
oA = V*oS*V';
oA = oA*0.5 + oA'*0.5;  %%to guarantee symmetric?
end
