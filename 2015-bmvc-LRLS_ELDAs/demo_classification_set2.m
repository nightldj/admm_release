close all;
clear;
clc;
addpath('.\lib\MinMaxSelection');
addpath('.\lib');

disp(' ')
disp('****************************')
disp('Low-rank Least Squares Exemplar-LDAs')
disp('In BMVC 2015')
disp('contact: xuzhustc@gmail.com')
disp('****************************')
disp(' ')

disp('demo: low rank least square exemplar-LDAs (LRLS-ELDAs), classification, office-caltech, D,W->A,C');

%% parameters
param.lda_shrink = 1; %L2 regularizer, \delta in the paper
param.low_rank_fac = 0.1; %trace norm regularizer, \xi in the paper
param.prdct_top_num = 5; %top number of exemplar classifier for fusion

%optimization parameters, 
param.lambda0 = 1; 
param.max_eig = 1e5;
param.step_size = 1;
param.max_ites = 100;
param.min_res = 1e-3;


%% load data
data_file = '.\office_caltech_dl.mat';
load(data_file, 'ms_data');
disp(['load from: ' data_file]);
disp(['domain: ' num2str(ms_data.dm_num) ', category: ' num2str(ms_data.cate_num) ', sample: ' num2str(ms_data.smpl_num)]);

%source domain: amazon, caltech, target domain: dslr, webcam
src_dm = [3 4];
disp(' ');
disp('src dm: ');
src_data = select_dm_data(ms_data, src_dm);
if isempty(src_data)
    disp(['warning: no src data, domain lbl : ' num2str(src_dm)]);
end
tgt_dm = [1 2];
disp(' ');
disp('tgt dm: ');
tgt_data = select_dm_data(ms_data, tgt_dm);
if isempty(tgt_data)
    disp(['warning: no tgt data, domain lbl : ' num2str(tgt_dm)]);
end

disp(' ')
disp(['start: ' num2str(clock)]);
disp(' ')

%% training
disp(' ');
disp('training...')
tid =tic;
model = lrls_elda_train(src_data, param, 'fasta');
time = toc(tid);
disp(['LRLSE-LDAs train time: ' num2str(time) ' s']);

%% testing
disp(' ');
disp('testing...')
tid = tic;
out_param = lrls_elda_predict(tgt_data.ftr, tgt_data.lbl, model);
time = toc(tid);
disp(['predict time : ' num2str(time) ' s ']);

disp(['ACC: ' num2str(out_param.accuracy)]);

disp(' ');
disp(['end: ' num2str(clock)]);
