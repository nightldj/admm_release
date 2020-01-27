close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo quadratic programming
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com


%% minimize  1/2 u'Qu+q'u s.t. Au<=b
% H(u) = 1/2 u'Qu+q'u, G(v) = Select(v<=b)
nu=500;
nb=250;

Q1 = randn(nu, nu);
Q=Q1'*Q1;

fprintf('QP, condition of Q:%e\n', cond(Q));
q = randn(nu, 1);
A = randn(nb, nu);
b = randn(nb, 1);

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
[sol1, outs1] = aradmm_QP(Q, q, A, b, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_QP(Q, q, A, b, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] = aradmm_QP(Q, q, A, b, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] = aradmm_QP(Q, q, A, b, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] = aradmm_QP(Q, q, A, b, opts);
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
hold off

