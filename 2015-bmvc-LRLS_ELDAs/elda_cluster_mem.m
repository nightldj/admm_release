function cls = elda_cluster_mem(pos, neg, w, param)
% spectral clustering based on exemplar classifiers, use sparse matrix to
% save memory
% input: pos.data, neg.data are n*d matrix, each row represent a data point
%        w: weight matrix, same size as pos.data, each row is the weight for the corresponding exemplar 
% ouput: 
%       cls: cluster idx for pos.data
% Zheng Xu, 2015   

pos_data = pos.data';
max_b = sum(w.*pos.data, 2);
neg_mu = mean(neg.data);
min_b = w*neg_mu';
%linear trasform for all positive: w*x
blk_size = param.blk_size;
smpl_num = size(pos_data, 2);
top_n = 10;
if smpl_num < 10000
    pre = w*pos_data;
    bias_max = repmat(max_b, [1 size(pos_data, 2)]);
    bias_min = repmat(min_b, [1 size(pos_data, 2)]);
    pre = (pre-bias_min)./max((bias_max-bias_min), eps('single'));
    prob = pre;
    prob (prob < 0) = 0;
    prob = (ones(size(prob))-eye(size(prob))).*prob;
else
    all_idx = zeros(size(w, 1), top_n);
    all_loc = zeros(size(w, 1), top_n);
    all_val = zeros(size(w, 1), top_n);
    blk_num = floor(smpl_num / blk_size);
    for blk_i = 1:blk_num
        w_idx = blk_size*(blk_i-1)+1:blk_size*blk_i;
        tmp_w = w(w_idx, :);
        tmp_pre = tmp_w*pos_data;
        tmp_prob = tmp_pre;
        tmp_prob(eye(size(tmp_prob)) > 0) = 0;
        [val, loc] = maxk(tmp_prob, top_n, 2);
        bias_max = repmat(max_b(w_idx), [1 top_n]);
        bias_min = repmat(min_b(w_idx), [1 top_n]);
        val = (val-bias_min)./max((bias_max-bias_min), eps('single'));
        val(val<0) = 0;
        idx = repmat(w_idx', 1, top_n);
        all_idx(w_idx, :) = idx;
        all_loc(w_idx, :) = loc;
        all_val(w_idx, :) = val;
    end
    if blk_size*blk_num < smpl_num
        w_idx = blk_size*blk_num+1:smpl_num;
        tmp_w = w(w_idx, :);
        tmp_pre = tmp_w*pos_data;
        tmp_prob = tmp_pre;
        tmp_prob(eye(size(tmp_prob)) > 0) = 0;
        [val, loc] = maxk(tmp_prob, top_n, 2);
        bias_max = repmat(max_b(w_idx), [1 top_n]);
        bias_min = repmat(min_b(w_idx), [1 top_n]);
        val = (val-bias_min)./max((bias_max-bias_min), eps('single'));
        val(val<0) = 0;
        idx = repmat(w_idx', 1, top_n);
        all_idx(w_idx, :) = idx;
        all_loc(w_idx, :) = loc;
        all_val(w_idx, :) = val;
    end
    prob = sparse(all_idx(:), all_loc(:), all_val(:));
end

aff = (prob + prob')/2;

%spectral clustering
cls = spectral_clustering(aff, pos.lbl_num, param.spectral_k);
end