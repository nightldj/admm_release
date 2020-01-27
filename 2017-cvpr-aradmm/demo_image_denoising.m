close all;
clear;
clc;
addpath('./solver')
%rng(2016);

% demo total vational image denoising
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  mu/2 ||x-f||^2 + lam1 |\grad x| 

%% load image
imgfile='cameraman256.tif';
img = double(imread(imgfile)); %load image
x_true = img;
fprintf('loaded image size: %d*%d*%d\n', size(x_true, 1), size(x_true, 2), size(x_true, 3));

sig = 20; % noise
x_noise = normrnd(0, sig, size(x_true));
x_given = x_true + x_noise;
x_given(x_given>255) = 255;
x_given(x_given<0) = 0;


%% model parameter
mu = 1;  %constraint
lam1 = 10; % l1 regularizer of gradient

%% solver parameter
opts.tol = 1e-3; %stop criterion, relative tolerance
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
[sol1, outs1] = aradmm_image_denoising(x_given, mu, lam1, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% relaxed ADMM
opts.adp_flag = 0;
opts.gamma = 1.5;
[sol2,outs2] =  aradmm_image_denoising(x_given, mu, lam1, opts);
t2 = outs2.runtime;
fprintf('relaxed ADMM complete after %d iterations!\n', outs2.iter);

% residual balancing
opts.adp_flag = 3; %residual balance
opts.gamma = 1;
[sol3,outs3] =  aradmm_image_denoising(x_given, mu, lam1, opts);
t3 = outs2.runtime;
fprintf('RB ADMM complete after %d iterations!\n', outs3.iter);

% Adaptive ADMM, AISTATS 2017
tic;
opts.adp_flag = 1;
opts.gamma = 1;
[sol4,outs4] =  aradmm_image_denoising(x_given, mu, lam1, opts);
t4 = outs4.runtime;
fprintf('adaptive ADMM complete after %d iterations!\n', outs4.iter);

% ARADMM
tic;
opts.adp_flag = 5; 
opts.gamma = 1;
[sol6,outs6] =  aradmm_image_denoising(x_given, mu, lam1, opts);
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

