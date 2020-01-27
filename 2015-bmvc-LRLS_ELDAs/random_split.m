function [pos, neg] = random_split(data, lbl, lbl_num)
% input: data are n*d matrix, each row represent a data point
%        lbl: groundtruth label of data
%        lbl_num: number of labels
% ouput: 
%       pos, neg: data split based on groundtruth labels
% Zheng Xu, 2015 
[data_num, data_dim]=size(data);
pos_lbl_num = ceil(lbl_num/2.0);
perm = randperm(lbl_num);
slct_lbl = perm(1:pos_lbl_num);
disp(['split: positive label: ' num2str(pos_lbl_num)]);
disp(slct_lbl);
idx = false(data_num, 1);
for i=1:pos_lbl_num
    idx = idx | (lbl == slct_lbl(i));
end
%positive
pos.data = data(idx, :);
pos.lbl = lbl(idx, :);
pos.lbl_num = pos_lbl_num;
pos.data_num = sum(idx);
pos.data_dim = data_dim;
pos.slct_lbl = slct_lbl;
pos.slct_idx = idx;
%negative
n_idx = ~idx;
neg.data = data(n_idx, :);
neg.lbl = data(n_idx, :);
neg.lbl_num = lbl_num - pos.lbl_num;
neg.data_num = sum(n_idx);
neg.data_dim = data_dim;
neg.slct_lbl = perm(pos_lbl_num+1:end);
neg.slct_idx = n_idx;
%refine positive label:
new_lbl = zeros(pos.data_num, 1);
for i = 1:pos_lbl_num
    new_lbl(pos.lbl == slct_lbl(i)) = i;
end
pos.lbl = new_lbl;
end