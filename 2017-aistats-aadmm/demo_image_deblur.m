%demo  ADMM for image deblur
% @author: Zheng Xu (xuzhustc@gmail.com), Apr. 2016
close all;
clear;
clc;
addpath('./solver')
rng(2016);


%% problem definition minimize  mu/2 ||K*x-f||^2 + lam1 |\grad x| + lam2/2 ||\grad x||^2
imgfile='cameraman256.tif';
img = double(imread(imgfile)); %load image
x_true = img;
fprintf('loaded image size: %d*%d*%d\n', size(x_true, 1), size(x_true, 2), size(x_true, 3));

b_sig = 2; %sigma for the blur kernel
gk = fspecial('gaussian', size(x_true), b_sig);
bkernel = zeros(size(x_true));
halfx = size(bkernel, 1)/2;
halfy = size(bkernel, 2)/2;
bkernel(1:halfx, 1:halfy) = gk(end-halfx+1:end, end-halfy+1:end);
bkernel(halfx+1:end, halfy+1:end) = gk(1:halfx, 1:halfy);
bkernel(halfx+1:end, 1:halfy) = gk(1:halfx, halfy+1:end);
bkernel(1:halfx, halfy+1:end) = gk(halfx+1:end, 1:halfy);
fbkernel = fft2(bkernel);
x_blur = real(ifft2(fbkernel.*fft2(x_true)));

sig = 5; % noise
x_noise = normrnd(0, sig, size(x_true));
x_given = x_blur + x_noise;
x_given(x_given>255) = 255;
x_given(x_given<0) = 0;

mu = 10;  %constraint
lam1 = 1; % l1 regularizer of gradient
lam2 = 0.1; % l2 regularizer of gradient


%% paramters
opts = get_default_opts();
opts.maxiter = 500; %max iteration
opts.tau = mu/10; %initial stepsize, follow Goldstein 14
opts.tol = 1e-3; %stop criterion, relative tolerance
opts.verbose = 0; 


fprintf('ADMM start...\n');
%%
% ADMM
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts);
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
opts.adp_flag = 5; %AADMM with spectral penalty
[sol2,outs2] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts);
fprintf('adaptive ADMM complete after %d iterations!\n', outs2.iter);

% Nesterov ADMM
opts.adp_flag = 2; % Nesterove ADMM
[sol3,outs3] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts);
fprintf('Nesterove ADMM complete after %d iterations!\n', outs3.iter);

% adaptive ADMM baseline: residual balance
opts.adp_flag = 3; %residual balance
[sol4,outs4] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts);
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline: normalized residual balance
opts.adp_flag = 4; %normalized residual balance
[sol5, outs5] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts);
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