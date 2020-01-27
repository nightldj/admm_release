close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo consensus problem: sparse logistic regression
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  \sum log(1+exp(-cDx)) + lam1 |x| 

%% problem defination
N = 2; %number of process
n = 1000; %sample number
fd = 20; %feature dimension
nd = 5; %uselsess features

%synthetic data by two Gaussian, last parameter controls the distance of
%the Gaussian, closer when it is larger, i.e., less separable by linear
%classifier
[D, c] = create_classification_problem(n, fd, nd, 1);

%% model parameter
lam1 = 1;

%% solver parameter
opts.tol = 1e-5; %stop criterion, relative tolerance
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
[sol1, outs1] = aradmm_logreg_consensus(D, c, N, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_logreg_consensus(D, c, N, lam1, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] =  aradmm_logreg_consensus(D, c, N, lam1, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] =  aradmm_logreg_consensus(D, c, N, lam1, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] =  aradmm_logreg_consensus(D, c, N, lam1, opts);
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
