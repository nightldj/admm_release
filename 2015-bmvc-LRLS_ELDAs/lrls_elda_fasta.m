function [w, outs] = lrls_elda_fasta(pos, neg, w0, params)
%FASTA solver for low-rank least squares exemplar-LDAs, based on forward
%backward splitting, http://www.cs.umd.edu/~tomg/FASTA.html
% input: pos.data, neg.data are n*d matrix, each row represent a data point
%        w0: initial weight matrix W
% ouput: 
%       w: weight matrix, same size as pos.data, each row is the weight for the corresponding exemplar 
% Zheng Xu, 2015 

tid = tic; % time recording
tau = params.step_size;
max_ites = params.max_ites;
max_eig = params.max_eig;
min_res = params.min_res;
delta = params.lda_shrink;
xi = params.low_rank_fac;
res0 = 1e10;

%initialize
pos_data = pos.data';
neg_data = neg.data';
neg_mu = mean(neg_data, 2);
[data_dim, pos_num] = size(pos_data);
[data_dim, neg_num] = size(neg_data);
pos_y = pos_data-repmat(neg_mu, [1 pos_num]);
neg_y = neg_data-repmat(neg_mu, [1 neg_num]);

%factor to update w
neg_cov = neg_y*neg_y';
neg_cov = neg_cov+delta*eye(data_dim);

opts.maxIters = max_ites;
opts.tol = min_res;
[sol, fouts] = fasta(@(x) x, @(x) x, @(w) fun_f(w, neg_y, pos_y, delta) ,  ...
    @(w)(neg_cov*w - pos_data), @(w) fun_g(w, xi) ,@(w, t) prox_lr(w, t, xi, max_eig), w0', opts);

w = sol';
outs.res = fouts.residuals;
outs.time = toc(tid);
disp(['lrls_elda_fasta: max ite:' num2str(length(fouts.residuals))]);
disp(['elapsed time(seconds):' num2str(outs.time)]);
end

function val = fun_f(w, neg_y, pos_y, delta)
tmp = (neg_y'*w).^2;
val = sum(tmp(:))/2.0;
tmp = w.^2;
val = val + delta*sum(tmp(:))/2.0;
tmp = w.*pos_y;
val = val - sum(tmp(:));
end


function val = fun_g(w, xi)
[f_u, f_s, f_v] = svd(w, 'econ');
val = sum(diag(f_s))*xi;
end

function new_w = prox_lr(w, t, xi, max_eig) 
   [f_u, f_s, f_v] = svd(w, 'econ');
    f_s_vec = diag(f_s);
    f_s_vec(f_s_vec > max_eig) = max_eig;  %%% constrain extreme large eigen value
    abs_s_vec = abs(f_s_vec);
    sh_f_s_vec = f_s_vec./max(abs_s_vec, eps('single')).*max(abs_s_vec-t*xi, 0); %shrink by xi*tau
    sh_f_s = diag(sh_f_s_vec);
    new_w = f_u*sh_f_s*f_v';
end
