function out_param = lrls_elda_predict(test_ftr, test_lbl, model)
%classification model testing with LRLSE-LDAs
% Zheng Xu, 2015 

predict_val = get_predict_val(test_ftr, model);
out_param = xmy_accuracy(predict_val, test_lbl);
out_param.model = model;
end

function predict_val = get_predict_val(test_ftr, model)
min_val = double(eps('single'));

[smpl_num, ftr_dim] = size(test_ftr);
cate_num = model.cate_num;

tid = tic;
esvm_predict_val = model.esvm.esvm_weights * test_ftr';
bias_min = repmat(model.esvm.esvm_bias_min, 1, smpl_num);
bias_max = repmat(model.esvm.esvm_bias_max, 1, smpl_num);
time = toc(tid);
disp(['exemplar predict time : ' num2str(time) ' s ']);

esvm_predict_p = (esvm_predict_val-bias_min)./(bias_max-bias_min);

predict_val = zeros(smpl_num, model.cate_num);
for cate_i = 1:cate_num
    disp(['predict cate: ' num2str(cate_i)]);
    cate_idx = (model.esvm.train_lbl == cate_i);
    if sum(cate_idx) > model.prdct_top_num
        top_flag = true;
    else
        top_flag = false;
    end
    for smpl_i = 1:smpl_num
        tmp_p = esvm_predict_p(cate_idx, smpl_i);
        if top_flag
            [top_p, top_loc] = maxk(tmp_p, model.prdct_top_num);
        else
            top_p = tmp_p;
        end
        predict_val(smpl_i, cate_i) = sum(top_p);
    end
end
end

function out_param = xmy_accuracy(predict_val, tst_lbl)
[smpl_num, cate_num] = size(predict_val);
assert(length(tst_lbl) == smpl_num);
[predict_max, predict_lbl] = max(predict_val, [], 2);
out_param.cate_conf_mat = zeros(cate_num);
for cate_i = 1:cate_num
    cate_idx = (tst_lbl == cate_i);
    for cate_j = 1:cate_num
        out_param.cate_conf_mat(cate_i,cate_j) = sum(predict_lbl(cate_idx) == cate_j)/sum(cate_idx);
    end
end
out_param.accuracy =  mean(diag(out_param.cate_conf_mat));
out_param.predict_lbl = predict_lbl;
end
