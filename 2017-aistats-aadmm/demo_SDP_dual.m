%demo ADMM
%dual form of SDP, reference: Zaiwen Wen paper
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  <C, X> st A(X) = b, X>=0
%dual min -b'y st A^*(y) + S = C, S >=0
%y -> u, S -> v, X -> l
% min -b'u st A*u + v = C, v>=0

%% synthetic data
n = 100;
r = 10;
m = 100;
C = randn(n);
C = (C + C')*0.5;
[V, D] = eig(C);
C = V*(abs(D) + 0.1*eye(n))*V';
x_true = randn(n,r);
x_true = x_true*x_true'/n;
A = zeros(m, n*n);
for i=1:m
    tmp = randn(n);
    A(i, :) = reshape((tmp+tmp')*0.5, 1, n*n);
end
b = A*x_true(:);


%% paramters
opts = get_default_opts();
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration, need opts.obj
%3: print out for adaptive ADMM
opts.verbose = 0; %
opts.tol = 1e-3; %relative tolerance
opts.maxiter = 2000; %max interation
opts.tau = 0.1; %initial stepsize


fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_sdp_dual(C, A, b, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_sdp_dual(C, A, b, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_sdp_dual(C, A, b, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_sdp_dual(C, A, b, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_sdp_dual(C, A, b, opts);
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