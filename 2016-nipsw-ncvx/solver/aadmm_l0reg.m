% demo 
% use the AADMM solver
% details in An Empirical Study of ADMM for Nonconvex Problems
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_l0reg(M, f, lam1, opts)

%% minimize  lam1 |x|_0 + 1/2 ||Mx-f||^2
[n, m] = size(M);
%1/2 |Mx-f|^2
h = @(x) norm(M*x-f)^2/2;  %regression term
%lam1 |x|_0
g = @(x) lam1*sum(abs(x)>1e-20); %regularizer
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
% proximal: min |x|_0 + 1/(2t)|| x - z ||^2 
%  opt condition:  hard-thresholding
proxg = @(z,t) ( (z.^2-2*t) > 0 ).*z;
solvg = @(u, l, t) proxg(u-l/t, lam1/t); %update v
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
sol = sol.v;

end
