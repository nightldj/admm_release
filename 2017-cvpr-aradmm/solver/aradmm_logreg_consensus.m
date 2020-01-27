function [sol, outs] = aradmm_logreg_consensus(D, c, N, lam1, opts)

% demo consensus problem: sparse logistic regression
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  \sum log(1+exp(-Ax)) + lam1 |x| 
D = [D ones(length(c), 1)]; %homogeneous coordiantes
A = diag(c)*D;
[n, d] = size(A);

%h(x) = \sum log(1+exp(-Aw))
global warm_start_u0;
warm_start_u0 = randn(N*d, 1);
h = @(x) logreg_obj_consensus(x, A, N);
solvh =  @(v, l, t) solv_logreg_consensus(A, N, opts.tol/10, v, l, t);
% g(x) = lam1 |x| 
g = @(x) lam1*sum(abs(x));
shrink = @(x, t) sign(x).*max(abs(x) - t,0);
proxg = @(z, t) shrink(z, lam1*t);
solvg = @(u, l, t) proxg(consensus_mean(u, N, d)-consensus_mean(l, N, d)/t, 1.0/N/t);
%objective
obj = @(u, v) h(repmat(v, N, 1)) + g(v);


fA = @(x) x;
fAt = @(x) x;
fB = @(x) - repmat(x, N, 1);
fb = 0;

opts.obj = obj; %objective function, used when verbose


%% initialization
x0 = randn(d, 1);
l0 = ones(size(fB(x0)));

%%
% ADMM solver
tic;
[sol0, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol.weight = sol0.v(1:end-1);
sol.bias = sol0.v(end);
end

function u = solv_logreg_consensus(A, N, tol, v, l, t)
global warm_start_u0;
options = optimoptions(@fminunc,'Algorithm','quasi-newton', 'GradObj', 'on', 'MaxIter', 100, 'TolFun', tol, 'TolX', tol, 'Display','off');
u = fminunc(@(x) logreg_consensus(x, A, N, v, l, t), warm_start_u0, options);
% opts.Method = 'lbfgs';
% opts.Display = 'off';
% opts.MaxIter = 200;
% opts.Corr = 50;
% opts.optTol = tol;
% u = minFunc(@(x) logreg_consensus(x, A, N, v, l, t), warm_start_u0, opts);
warm_start_u0 = u;
end

function [obj, grad] = logreg_consensus(x, A, N, v, l, t)
obj = logreg_obj_consensus(x, A, N) + 0.5*t*norm(x-repmat(v, N, 1)-l/t, 2)^2;
grad = logreg_grad_consensus(x, A, N) + t*x - t*repmat(v, N ,1) - l;
end

function g = logreg_grad_consensus(x, A, N)
[n, d] = size(A);
nn = ceil(double(n)/double(N));
g = zeros(size(x));
for i=1:N
    if i == N
        xend = n - (N-1)*nn;
    else
        xend = nn;
    end
    g(((i-1)*d+1):i*d) = logreg_grad( x(((i-1)*d+1):i*d), A(((i-1)*nn+1):((i-1)*nn+xend), :)); 
end
end

function y = logreg_obj_consensus(x, A, N)
[n, d] = size(A);
nn = ceil(double(n)/double(N));
ys = zeros(N, 1);
for i=1:N
    if i == N
        xend = n - (N-1)*nn;
    else
        xend = nn;
    end
    ys(i) = logreg_fun( x(((i-1)*d+1):i*d), A(((i-1)*nn+1):((i-1)*nn+xend), :)); 
end
y=sum(ys);
end

function y = logreg_fun(x, A)
%\sum log(1+exp(-A*x))
z = A*x;
f = zeros(size(z));
idx = z<0;
f(~idx) = log(1+exp(-z(~idx)));
f(idx) = -z(idx)+log(1+exp(z(idx)));
y = sum(f);
end

function g = logreg_grad(x, A)
z = A*x;
f1 = zeros(size(z));
pz_ind = z>0;
f1(pz_ind) = -exp(-z(pz_ind))./(1+exp(-z(pz_ind)));
pz_ind = z<=0;
f1(pz_ind) = -1./(1+exp(z(pz_ind)));
g = A'*f1;
end

function z = consensus_mean(x, N, d)
zmat = reshape(x, d, N);
z = mean(zmat, 2);
end