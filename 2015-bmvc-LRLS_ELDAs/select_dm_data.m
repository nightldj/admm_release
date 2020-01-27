function dm_data = select_dm_data(ms_data, dm_ids)
%select data for classificatin
%Zheng Xu, 2015
dm_idx = false(size(ms_data.dm_lbl));
for d_i = 1:length(dm_ids)
    dm_idx(ms_data.dm_lbl == dm_ids(d_i)) = true;
end

if 0 == sum(dm_idx)
    disp(['empty domain : ' num2str(dm_ids)]);
end

ftr = ms_data.ftr(dm_idx, :);
lbl = ms_data.cate_lbl(dm_idx);

lbl_num = length(unique(lbl));
assert(max(lbl) == lbl_num);
[smpl_num, smpl_dim] = size(ftr);
assert(length(lbl) == smpl_num);

disp(['select domains: ' num2str(dm_ids) ', labels: ' num2str(lbl_num) ', samples: ' num2str(smpl_num) ', dim: ' num2str(smpl_dim)]);

dm_data.ftr = ftr;
dm_data.lbl = lbl;
dm_data.idx = dm_idx;
dm_data.dm_lbl = ms_data.dm_lbl(dm_idx);

dm_data.lbl_num = lbl_num;
dm_data.ftr_dim = ms_data.ftr_dim;
dm_data.smpl_num = smpl_num;
end