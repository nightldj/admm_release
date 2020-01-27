%demo  ADMM for image restoration
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize  mu/2 ||x-f||^2 + lam1 |\grad x| 
imgfile= 'cameraman256.tif';

img = double(imread(imgfile)); %load image
x_true = img;
fprintf('loaded image size: %d*%d*%d\n', size(x_true, 1), size(x_true, 2), size(x_true, 3));

mu = 1;%constraint
lam1 = 500; % l1 regularizer of gradient

sig = 20; %12 for uniform % noise
x_noise = normrnd(0, sig, size(x_true));
x_given = x_true + x_noise;
x_given(x_given>255) = 255;
x_given(x_given<0) = 0;



%% paramters
opts = get_default_opts();
opts.maxiter = 200; %max iteration
opts.tau = 0.1; %initial timestep
opts.tol = 0.5; %stop criterion, dres < tol*dres0 && pres < tol*pres0a
opts.verbose = 0; 


fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_imgres(x_given, mu, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_imgres(x_given, mu, lam1, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_imgres(x_given, mu, lam1, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_imgres(x_given, mu, lam1, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_imgres(x_given, mu, lam1, opts);
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