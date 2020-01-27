close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo low rank least squares
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  lam1 |W|_* + lam2/2 ||W||^2 + 1/2 ||D(W)-c||^2

%% define synthetic problem
% Exemplar LDA, Xu et al. BMVC 2015
np = 500;  %positive sample number
nn = 500;  %negative sample number
d = 200; %sample dimension
r = 20;

W = zeros(d, np);
W(1:d, 1:d) = randn(d, r)*randn(r, d);
D = randn(np+nn, d);
c = D*W + 0.1*randn(np+nn, np);

%% application model parameter
lam1 = 1;
lam2 = 1;


%% solver parameter
opts.tol = 1e-12; %stop criterion, relative tolerance
opts.maxiter = 200; %max interation
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
[sol1, outs1] = aradmm_lrls(D, c, lam1, lam2, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_lrls(D, c, lam1, lam2, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] = aradmm_lrls(D, c, lam1, lam2, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] = aradmm_lrls(D, c, lam1, lam2, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] = aradmm_lrls(D, c, lam1, lam2, opts);
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

