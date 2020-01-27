function [w, outs] = lrls_elda_admm(pos, neg, w0, l0, params)
%admm solver for low-rank least squares exemplar-LDAs
% input: pos.data, neg.data are n*d matrix, each row represent a data point
%        w0: initial weight matrix W
%        l0: initial dual variable matrix Lambda
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
p_res0 = 1e10;
d_res0 = 1e10;

%initialize
w=w0';
f=w;
new_f=f;
l=l0*ones(size(f));
pos_data = pos.data';
neg_data = neg.data';
neg_mu = mean(neg_data, 2);
[data_dim, pos_num] = size(pos_data);
[data_dim, neg_num] = size(neg_data);
pos_y = pos_data-repmat(neg_mu, [1 pos_num]);
neg_y = neg_data-repmat(neg_mu, [1 neg_num]);

%factor to update w
neg_cov = neg_y*neg_y';
neg_cov = neg_cov+(delta+tau)*eye(data_dim);

%invert by svd
[cov_u, cov_s, cov_v] = svd(neg_cov, 'econ');
cov_tmp = diag(cov_s);
cov_tmp = 1./cov_tmp;
cov_tmp(cov_tmp>max_eig) = 0;
inv_cov_s = diag(cov_tmp);
inv_cov = cov_v*inv_cov_s*cov_u';

%residual
p_res = zeros(max_ites, 1);
d_res = zeros(max_ites, 1);

for ite = 1:max_ites
    f = new_f;
    %update w
    w = inv_cov*(pos_y + tau*(f-l));
    %update f
    [f_u, f_s, f_v] = svd(w+l, 'econ');
    f_s_vec = diag(f_s);
    f_s_vec(f_s_vec > max_eig) = max_eig;  %%% constrain extreme large eigen value
    abs_s_vec = abs(f_s_vec);
    sh_f_s_vec = f_s_vec./(abs_s_vec+eps('single')).*max(abs_s_vec-xi/tau, 0); %shrink by xi/tau
    sh_f_s = diag(sh_f_s_vec);
    new_f = f_u*sh_f_s*f_v';
    %update lambda
    l = l + w-new_f;
    %residual
    res_w = w-new_f;
    res_f = f-new_f;
    p_resk = norm(res_w(:), 2);
    d_resk = norm(res_f(:), 2);
    p_res(ite) = p_resk;
    d_res(ite) = d_resk;
    if 1==ite
        p_res0 = p_resk;
        d_res0 = d_resk;
    end
    if p_resk < min_res*p_res0 && d_resk < min_res*d_res0
        disp(['lrls_elda_admm: converge at ite:' num2str(ite)]);
        break;
    end
end
w = w';
outs.p_res = p_res(1:ite);
outs.d_res = d_res(1:ite);
outs.time = toc(tid);
disp(['lrls_elda_admm: max ite:' num2str(max_ites)]);
disp(['elapsed time(seconds):' num2str(outs.time)]);
end

