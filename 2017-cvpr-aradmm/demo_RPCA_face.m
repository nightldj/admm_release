close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo Robust PCA
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com


%% minimize_{A, E} |A|_* + lam1 |E| st A+E=C
% face
load('face.mat')
p_scale = 255.0; %255.0
set1 = set1/p_scale;
[img_num, img_size] = size(set1);
fprintf('number of images: %d\n', img_num);
C = set1;
[n, m] = size(C);

lam1 = 0.05; %L1, sparse error

obj = @(A, E) rpca_obj(A, E, lam1);

solvh = @(v, l, t) prox_trace_norm(C-v+l/t, 1/t); %update u
% min lam1 |V|_* + lam2/2 ||V||_F^2 + t/2 ||l/t-W+V|| 
%  opt condition
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
solvg = @(u, l, t) shrink(C-u+l/t, lam1/t); %update v
fA = @(x) x;
fAt = @(x) x;
fB = @(x) x;
fb = C;

%% solver parameter
opts.tol = 1e-12; %stop criterion, relative tolerance
opts.maxiter = 1000; %max interation
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
[sol1, outs1] = aradmm_RPCA(C, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_RPCA(C, lam1, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] = aradmm_RPCA(C, lam1, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] = aradmm_RPCA(C, lam1, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] = aradmm_RPCA(C, lam1, opts);
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