%demo ADMM, RPCA
function demo
close all;
clear;
clc;
addpath('./solver')
rng(2016);

%% minimize_{A, E, Z} |A|_* + lam1 |E| + lam2*0.5*|C-A-E|_F  
n = 200;
r = 0.1*n;
A = randn(n, r)*randn(r, n); %low rank
A = A./max(abs(A(:)));
sig = 1;
E = sig*sprand(n, n, 0.05)-sig*sprand(n, n, 0.05); %sparse
C = A + E + 0.01*randn(n, n); %noisy
[n, m] = size(C);
lam2 = 1; %L2, regression
lam1 = lam2/10; %L1, sparse error

%% subproblem solver
% lam2*0.5*|u1|_F + lam1 |u2| + |u3|_* s.t. u1+u2+u3 = C   
obj = @(u) rpca_obj(u, lam1, lam2, C);

fA{1} = @(x) x;
fA{2} = @(x) x;
fA{3} = @(x) x;
fAt{1} = @(x) x;
fAt{2} = @(x) x;
fAt{3} = @(x) x;
fb = C(:);

% min lam2*0.5*||Z|| + t/2 || C-E-A-Z+L/t||
solvh{1} = @(u, l, t) (fb + l/t - u{2} - u{3}) *t /(lam2+t);
% min lam1*|E| + t/2 || C-E-A-Z+L/t||
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
solvh{2} = @(u, l, t) shrink(fb+l/t-u{1}-u{3}, lam1/t);
% min ||A||_* + t/2 || C-E-A-Z+L/t||
solvh{3} = @(u, l, t) prox_trace_norm(n, m, fb+l/t-u{1}-u{2}, 1/t);

%% initialization
u0{1} = 0.1*randn(n*m, 1); %noise
u0{2} = zeros(n*m, 1); %sparse
u0{3} = fb;  %low rank
l0 = ones(n*m, 1);


%% paramters
opts.tol = 1e-4; %stop criterion, relative tolerance
opts.maxiter = 2000; %max interation
opts.tau = 0.1; %initial stepsize
%verbose print
%0: no print, 
%1: print every iteration
%2: evaluate objective every iteration, need opts.obj
%3: print out for adaptive ADMM
opts.verbose = 0; % 
opts.obj = obj; %objective function, used when verbose
opts.adp_freq = 2; %frequency for adaptive step size
opts.adp_start_iter = 2; %start iteration of adaptive stepsize
opts.adp_end_iter = 1000; %end iteration of adaptive stepsize
opts.orthval = 0.2; %value to test orthogonal or not
opts.beta_scale = 2; %RB
opts.res_scale = 0.1; %RB


fprintf('ADMM start...\n');
%% Run experiments
% ADMM 
tic;
opts.adp_flag = 0; %fix tau, no adaptation
[sol1,outs1] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t1 = toc;
fprintf('vanilla ADMM complete after %d iterations!\n', outs1.iter);

% adaptive ADMM
tic;
opts.adp_flag = 1; %AADMM with spectral penalty
[sol2,outs2] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t2 = toc;
fprintf('Adaptive Multi-block ADMM (AMADMM) complete after %d iterations!\n', outs2.iter);


% adaptive ADMM baseline: residual balance
tic;
opts.adp_flag = 3; %residual balance
[sol4,outs4] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t4 = toc;
fprintf('RB ADMM complete after %d iterations!\n', outs4.iter);

% adaptive ADMM baseline
tic;
opts.adp_flag = 4;%normalized: residual balance
[sol5, outs5] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t5 = toc;
fprintf('NRB ADMM complete after %d iterations!\n', outs5.iter);

% approx adaptive ADMM
tic;
opts.adp_flag = 5;%normalized: residual balance
[sol6, outs6] = amadmm_core(solvh, fA, fAt, fb, u0, l0, opts);
t6 = toc;
fprintf('Approx AMADMM complete after %d iterations!\n', outs6.iter);


legends = {'Vanilla ADMM', 'Residual balance', 'Normalized RB', 'Approx AADMM', 'Adaptive ADMM'};
figure, 
semilogy(outs1.tols, '-.g'), 
hold, 
semilogy(outs4.tols, '--m');
semilogy(outs5.tols, '--', 'Color',[0.7 0.2 0.2]);
semilogy(outs6.tols, 'c');
semilogy(outs2.tols, 'b');
ylabel('Relative residual', 'FontName','Times New Roman'); 
xlabel('Iteration', 'FontName','Times New Roman');
ylim([10^(-4) 10]);
legend(legends, 'FontName','Times New Roman');

end

function obj = rpca_obj(u, lam1, lam2, C)
%lam2*0.5*|u1|_F + lam1 |u2| + |u3|_* s.t. u1+u2+u3 = C
A = reshape(u{3}, size(C));
[~,S,~] = svd(A,'econ');
s = diag(S);
obj = 0.5*lam2*norm(C(:)-u{2}-u{3}, 2) + lam1 * norm(u{2}, 1) + sum(abs(s)) ;
end

function u1 = prox_trace_norm(n, m, vec, t)
% min ||U||_* + 0.5/t ||U-A||_F
A = reshape(vec, n, m);
[U,S,V] = svd(A,'econ');
s = diag(S);
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
os = shrink(s, t);
oS = diag(os);
oA = U*oS*V';
u1 = oA(:);
end