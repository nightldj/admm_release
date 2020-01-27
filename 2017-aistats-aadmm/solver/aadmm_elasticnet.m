% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_elasticnet(M, f, lam1, lam2, opts)

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
    % [U, S, V] = svd(M); %M = USV'
    % diagS2 = diag(S'*S);
    % fprintf('svd complete!');
    % proxh = @(z,t) V*(diag(1./(t*diagS2+1)))*V'*(z+t*Mf);
    M2 = M*M';
    proxh = @(z, t) (z+t*Mf) - t*M'*((t*M2+speye(n))\(M*(z+t*Mf)));  %Woodbury Matrix Identity
else
    M2 = M'*M;
    proxh = @(z,t) (t*M2+speye(m))\(z+t*Mf);
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


tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
end