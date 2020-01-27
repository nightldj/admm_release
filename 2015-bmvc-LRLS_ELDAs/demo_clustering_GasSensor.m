close all;
clear;
clc;
addpath('.\lib');
%run('D:\Code\matlab\lib\vlfeat-0.9.19\toolbox\vl_setup'); %results in paper use vlfeat for clustering, modify it in spectral_clustering.m

disp(' ')
disp('****************************')
disp('Low-rank Least Squares Exemplar-LDAs')
disp('In BMVC 2015')
disp('contact: xuzhustc@gmail.com')
disp('****************************')
disp(' ')

disp('demo: low rank least square exemplar-LDAs (LRLS-ELDAs), clustering, gas sensor');

%% parameters
rep_num = 5;  %repeat number of experiments, 50 in the paper
param.lda_shrink = 1; %L2 regularizer, \delta in the paper
param.low_rank_fac = 0.1; %trace norm regularizer, \xi in the paper
param.spectral_k = 10; %eigen vectors used for spectral clustering
param.blk_size = 1000; %block size for building the affinity matrix for spectral clustering

%optimization parameters, 
param.lambda0 = 1; 
param.max_eig = 1e5;
param.step_size = 1;
param.max_ites = 100;
param.min_res = 1e-3;

%% load data
load('.\GasSensor.mat');
assert(max(lbl) == lbl_num);

pur_vec = zeros(rep_num, 1);
for j=1:rep_num
    disp(['***repeat: ' num2str(j)]);
    [pos, neg] = random_split(data, lbl, lbl_num);
    w0 = rand(size(pos.data));
    w = lrls_elda_admm(pos, neg, w0, param.lambda0, param);
    %w = lrls_elda_fasta(pos, neg, w0, param);
    cls = elda_cluster_mem(pos, neg, w, param);
    [miss, index] = missclassGroups(cls, pos.lbl, pos.lbl_num, 'fast');
    pur_vec(j) = double(pos.data_num-miss)/double(pos.data_num);
    disp(['purity:' num2str(pur_vec(j))]);
end
disp('************* results ***************')
disp(['purity mean: ' num2str(100*mean(pur_vec))]);
disp(['purity std: ' num2str(100*std(pur_vec)/sqrt(rep_num))]);