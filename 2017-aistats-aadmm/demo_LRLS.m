%demo ADMM
close all;
clear;
clc;
addpath('./solver')
rng(2016);


%% minimize  lam1 |W|_* + lam2/2 ||W||^2 + 1/2 ||D(W)-c||^2
% synthetic problem
lam1 = 1;
lam2 = 1;

np = 500;
nn = 500;
d = 200;
r = 20;

W = zeros(d, np);
W(1:d, 1:d) = randn(d, r)*randn(r, d);
D = randn(np+nn, d);
c = D*W + 0.1*randn(np+nn, np);


%% paramters
opts = get_default_opts();
opts.verbose = 0; 
opts.tol = 1e-5;
opts.maxiter = 2000; %max iteration
opts.tau = 0.1; %initial stepsize



fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_lrls(D, c, np, lam1, lam2, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_lrls(D, c, np, lam1, lam2, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_lrls(D, c, np, lam1, lam2, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_lrls(D, c, np, lam1, lam2, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_lrls(D, c, np, lam1, lam2, opts);
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