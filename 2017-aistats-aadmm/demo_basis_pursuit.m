close all;
clear;
clc;
addpath('./solver')
rng(2016);

% demo ADMM, basis pursuit
% use AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter Selection
% AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com
% reference: Boyd, http://web.stanford.edu/~boyd/papers/admm/

%% minimize  |x| s.t. Ax = b
% H(u) = Select(Au=b), G(v) = |v|
n = 30;
m = 10;
A = randn(m,n);
x_true = sprandn(n, 1, 0.1)*1;
b = A*x_true;

%% paramters
opts = get_default_opts();
opts.maxiter = 300; %max iteration
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
[sol1,outs1] = aadmm_basis_pursuit(A, b, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_basis_pursuit(A, b, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_basis_pursuit(A, b, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_basis_pursuit(A, b, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_basis_pursuit(A, b, opts);
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
xlabel('Iteration', 'FontName','Times New Roman');
ylim([10^(-5) 10]);
legend(legends, 'FontName','Times New Roman');