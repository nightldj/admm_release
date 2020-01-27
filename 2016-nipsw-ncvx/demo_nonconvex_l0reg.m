%demo ADMM, nonconvex, l0 regularizer
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  lam1 |x|_0 + 1/2 ||Mx-f||^2
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

lam1 = 1;

%% paramters
opts = get_default_opts();
opts.maxiter = 2000; %max iteration
opts.tau = .1; %initial stepsize
opts.tol = 1e-5; %relevant stop criterion
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration, in out.obj
%3: more print out for debugging adaptive relaxed ADMM
opts.verbose = 0; 


fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_l0reg(M, f, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_l0reg(M, f, lam1, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_l0reg(M, f, lam1, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_l0reg(M, f, lam1, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_l0reg(M, f, lam1, opts);
fprintf('NRB ADMM complete after %d iterations!\n', outs5.iter);

%%
legends = {'Vanilla ADMM', 'Fast ADMM', 'Residual balance', 'Normalized RB', 'Adaptive ADMM'};
figure,
semilogy(outs1.tols, '-.g'),
hold,
semilogy(outs3.tols, '-.r');
semilogy(outs4.tols, '--m');
semilogy(outs5.tols, '--', 'Color',[0.7 0.2 0.2]);
semilogy(outs2.tols, 'b');
ylabel('Relative residual', 'FontName','Times New Roman');
ylim([10^(-3) 10]);
xlabel('Iteration', 'FontName','Times New Roman');
legend(legends, 'FontName','Times New Roman');