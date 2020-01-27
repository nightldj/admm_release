close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo linear regression with elastic net regularizer
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  lam1 |x| + lam2/2 ||x||^2 + 1/2 ||Mx-f||^2

%% define synthetic problem 
% data matrix M, measurement f
n=50; %sample number
m=40; %sample dimension
sig_e = 1; 
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


%% solver parameter
opts.tol = 1e-12; %stop criterion, relative tolerance
opts.maxiter = 2000; %max interation
opts.tau = 0.1; %initial stepsize
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration
%3: more print out for debugging adaptive relaxed ADMM
opts.verbose = 0; 
opts = get_default_opts(opts);


fprintf('ADMM start...\n');
%%

% vanilla ADMM
opts.adp_flag = 0;
opts.gamma = 1;
[sol1, outs1] = aradmm_elasticnet(M, f, lam1, lam2, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_elasticnet(M, f, lam1, lam2, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] =  aradmm_elasticnet(M, f, lam1, lam2, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] =  aradmm_elasticnet(M, f, lam1, lam2, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] =  aradmm_elasticnet(M, f, lam1, lam2, opts);
t6 = outs6.runtime;
fprintf('ARADMM complete after %d iterations!\n', outs6.iter);

%%

legends = {'Vanilla ADMM', 'Relaxed ADMM', 'Residual balance', 'Adaptive ADMM', 'ARADMM'};


figure,
semilogy(outs1.tols, '-.g'),
hold,
semilogy(outs2.tols, '-.r');
semilogy(outs3.tols, '--m');
semilogy(outs4.tols, '--', 'Color',[0.7 0.2 0.2]);
semilogy(outs6.tols, 'b');
ylabel('Relative residual', 'FontName','Times New Roman');
xlabel('Iteration', 'FontName','Times New Roman');
legend(legends, 'FontName','Times New Roman');

