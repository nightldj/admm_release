%demo ADMM, SVM, dual, QP
%reference: LibSVM
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  1/2 ||x||^2 + C sum(max{1-Ax, 0})
%dual min 1/2 a'Qa - 1'a s.t. y'a =0 0<=a<=c
% ADMM min 1/2 a'Qa - 1'a + select(b~[0,C]) s.t. y'a =0 a=b
n = 1000; %sample number
fd = 20; %feature dimension
nd = 5; %uselsess features

%synthetic data by two Gaussian, last parameter controls the distance of
%the Gaussian, closer when it is larger, i.e., less separable by linear
%classifier
[D, lbl] = create_classification_problem(n, fd, nd, 1);

svm_C = 0.1; %hinge loss

%% paramters
opts = get_default_opts();
opts.tol = 1e-3; %stop criterion, relative tolerance
opts.maxiter = 2000; %max interation
opts.tau = 0.1; %initial stepsize
opts.verbose = 0;

fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_svm_dual(D, lbl, svm_C, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_svm_dual(D, lbl, svm_C, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_svm_dual(D, lbl, svm_C, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_svm_dual(D, lbl, svm_C, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_svm_dual(D, lbl, svm_C, opts);
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