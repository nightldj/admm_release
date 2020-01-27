%demo multiblock ADMM, elastic net
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  lam1 |x| + lam2/2 ||x||^2 + 1/2 ||Mx-f||^2
n=50; %sample number
m=40; %dimension
sig_e = 1;  % 1 or 0.1, harder for 0.1
v1 = randn(n, 1);
v2 = randn(n, 1);
v3 = randn(n, 1);
M = zeros(n, 40);
E = normrnd(0, sig_e, n, 15);
M(:, 1:5) = repmat(v1, 1, 5) ;
M(:, 6:10) = repmat(v2, 1, 5);
M(:, 11:15) = repmat(v3, 1, 5);
M(:, 1:15) = M(:, 1:15) + E;
M(:, 16:40) = randn(n, 25);
x_true = zeros(m, 1);
x_true(1:15) = 3;
%x_true = x_true*100;
eta = normrnd(0, 0.1, n, 1);
%M = M*100;
f = M*x_true + eta;

%% application parameter
lam1 = 1; %L1 regularizer
lam2= 1; %L2 regularizer


%% subproblem solver
%1/2 |Mx-f|^2
h = @(x) norm(M*x-f)^2/2;  %regression term
%lam1 |x| + lam2/2 |x|^2
g = @(x) lam1*sum(abs(x)); %l1 regularizer
h3 = @(x) x'*x*lam2/2; %l2 regularizer
obj = @(u) h(u{3})+g(u{3})+h3(u{3});

fA{1} = @(x) [x; x];
fA{2} = @(x) [-x; zeros(size(x))];
fA{3} = @(x) [ zeros(size(x)); -x];
fAt{1} = @(x) x(1:m) + x(m+1:2*m);
fAt{2} = @(x) -x(1:m);
fAt{3} = @(x) -x(m+1:2*m);
fb = 0;
Mf = M'*f;
if n<m % M is fat
    M2 = M*M';
    solvh{1} = @(u, l, t) (fAt{1}(l)/t + u{2} + u{3} + Mf/t)*0.5 - M'*((speye(n)+M2*0.5/t)\(M*(fAt{1}(l)/t + u{2} + u{3} + Mf/t)*0.25/t));  %Woodbury Matrix Identity
else
    M2 = M'*M;
    solvh{1} = @(u, l, t) (M2+2*t*speye(m)) \ (fAt{1}(l) + t*(u{2} + u{3}) + Mf); %update u
end
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
solvh{2} = @(u, l, t) shrink(u{1}+fAt{2}(l)/t, lam1/t);
solvh{3} = @(u, l, t) (t*u{1}+fAt{3}(l))/(lam2+t);

%% initialization
u0{1} = randn(m,1);
u0{2} = u0{1};
u0{3} = u0{1};
l0 = ones(2*m, 1);


%% paramters
opts.maxiter = 2000; %max iteration
opts.tau = .1; %initial stepsize
opts.tol = 1e-6; %relevant stop criterion
%verbose print
%0: no print, 
%1: print every iteration
%2: evaluate objective every iteration, need opts.obj
%3: print out for adaptive ADMM
opts.verbose = 0; % 
opts.obj = obj; %objective function, used when verbose
opts.adp_freq = 2; %frequency for adaptive step size
opts.adp_start_iter = 2; %start iteration of adaptive stepsize
opts.adp_end_iter = 1000; %end iteration of adaptive stepsize
opts.orthval = 0.2; %value to test orthogonal or not
opts.beta_scale = 2; %RB
opts.res_scale = 0.1; %RB


fprintf('ADMM start...\n');
%% Run experiments
% ADMM 
tic;
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t1 = toc;
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
tic;
opts.adp_flag = 1; %AADMM with spectral penalty
[sol2,outs2] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t2 = toc;
fprintf('Adaptive Multi-block ADMM (AMADMM) complete after %d iterations!\n', outs2.iter);


% adaptive ADMM baseline: residual balance
tic;
opts.adp_flag = 3; %residual balance
[sol4,outs4] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t4 = toc;
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline
tic;
opts.adp_flag = 4;%normalized: residual balance
[sol5, outs5] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t5 = toc;
fprintf('NRB ADMM complete after %d iterations!\n', outs5.iter);

% approx adaptive ADMM
tic;
opts.adp_flag = 5;%normalized: residual balance
[sol6, outs6] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t6 = toc;
fprintf('Approx AMADMM complete after %d iterations!\n', outs6.iter);


legends = {'Vanilla ADMM', 'Residual balance', 'Normalized RB', 'Approx AADMM', 'Adaptive ADMM'};
figure, 
semilogy(outs1.tols, '-.g'), 
hold, 
semilogy(outs4.tols, '--m');
semilogy(outs5.tols, '--', 'Color',[0.7 0.2 0.2]);
semilogy(outs6.tols, 'c');
semilogy(outs2.tols, 'b');
ylabel('Relative residual', 'FontName','Times New Roman'); 
xlabel('Iteration', 'FontName','Times New Roman');
ylim([10^(-6) 10]);
legend(legends, 'FontName','Times New Roman');
