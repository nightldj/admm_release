function [cls, lap, nlap] = spectral_clustering(aff, lbl_num, spectral_k)
% input: aff: affinity matrix, diagnal should be zeros
%        lbl_num: number of clusters
%        spectral_l: number of eigen vectors used in spectral clustering
% ouput: 
%       cls: cluster idx for pos.data
%       lap: laplacian matrix
%       nlap: normalized laplacian matrix
% Zheng Xu, 2015  

assert(sum(diag(aff)) == 0);

%laplacian matrix
sum_aff = full(sum(aff));
lap = aff; %formulation in Ng, Jordan, Weiss Paper

%spectral clustering
sum_aff2 = 1./(sqrt(sum_aff)+eps('single'));
nlap=sum_aff2'*sum_aff2.*lap;
if size(nlap, 1) > spectral_k
    try
        [lap_v, ~]=eigs(double(nlap), spectral_k);
        lap_x = lap_v(:, 1:spectral_k);
    catch
        lap_x = nlap;
    end
else
    [lap_v, ~]=eig(double(nlap));
    lap_x = lap_v;
end
div = max(sqrt(sum(lap_x.^2, 2)), eps('single'));
lap_nx = lap_x./repmat(div, [1, size(lap_x, 2)]);
%[~, cls] = vl_kmeans(lap_nx', lbl_num, 'NumRepetitions', 5, 'Algorithm', 'ELKAN'); %need vlfeat
cls = kmeans(lap_nx, lbl_num, 'Replicates', 5);
end