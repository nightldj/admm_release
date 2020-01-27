function model = lrls_elda_train(data, param, opt)
%classification model training with LRLSE-LDAs
% input: data,
%        param,
%        opt, 'admm'/'fasta'
% ouput: 
%       model
% Zheng Xu, 2015 

train_ftr = data.ftr;
train_lbl = data.lbl;
[trn_smpl_num, ftr_dim] = size(train_ftr);
cate_num = max(train_lbl);

weights = zeros(size(train_ftr));
bias_min = zeros(trn_smpl_num, 1);
bias_max = zeros(trn_smpl_num, 1);
param.cate_num = cate_num;
for cate_i = 1:cate_num
    disp(['LRLSE-LDAs training... : category : ' num2str(cate_i)]);
    param.cate_i = cate_i;
    param.cate_j = 0;
    cate_idx = (train_lbl == cate_i);
    pos.data = train_ftr(cate_idx, :);
    neg.data = train_ftr(~cate_idx, :);
    
    w0 = rand(size(pos.data));
    if strcmp(opt, 'admm') == 1
        w = lrls_elda_admm(pos, neg, w0, param.lambda0, param);
    elseif strcmp(opt, 'fasta') == 1
        w = lrls_elda_fasta(pos, neg, w0, param);
    else
        disp('unknown opt in classifier training');
    end
    
    weights(cate_idx, :) = w;
    neg_mu = mean(neg.data);
    bias_min(cate_idx) = w*neg_mu';
    bias_max(cate_idx) = sum(w.*pos.data, 2);
end
model.esvm.esvm_weights = weights;
model.esvm.esvm_bias_min = bias_min;
model.esvm.esvm_bias_max = bias_max;
model.esvm.train_lbl = train_lbl;
model.cate_num = cate_num;
model.trn_smpl_num = trn_smpl_num;
model.prdct_top_num = param.prdct_top_num;
end
