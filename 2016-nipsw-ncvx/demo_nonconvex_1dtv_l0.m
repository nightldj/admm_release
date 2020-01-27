%demo  ADMM for total variation
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  mu/2 ||x-f||^2 + lam1 |\grad x|_0
n = 100;
x0 = ones(n,1);
for j = 1:3
    idx = randsample(n,1);
    k = randsample(2:5,1);
    x0(ceil(idx/2):idx) = k*x0(ceil(idx/2):idx);
end
x_given = x0 + 0.5*randn(n, 1);
x_true = x0;
mu = 1;
lam1 = 1; % L1 regularizer


%% paramters
opts = get_default_opts();
opts.maxiter = 2000; %max iteration
opts.tau = 0.1; %initial timestep
opts.tol = 1e-5; %stop criterion, dres < tol*dres0 && pres < tol*pres0a
opts.verbose = 0;


fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_tvl0(x_given, mu, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_tvl0(x_given, mu, lam1, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_tvl0(x_given, mu, lam1, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_tvl0(x_given, mu, lam1, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_tvl0(x_given, mu, lam1, opts);
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