function [D, c] = create_classification_problem(Nd, Nf, Nnf, h)
% The function generates Nd data vectors, each containing Nf features. Half the points
% should be in class 1 the others should be in class -1. The two classes of points should
% be approximately linearly separable. The parameter h determines how hard/easy it is to
% separate the points. When h=0, this classes should be easy to separate using a plane.
% When h is large, the problem becomes very difficult to separate with a plane.
% The function returns a feature matrix D with Nd rows and Nf columns. It also returns a
% column vector c containing the class labels of the corresponding feature vectors.

%generate by two Gaussians, h is larger, the centers are closer
c1 = zeros(Nf, 1);
r = 1;
dist = 10*r/(1+h);
c2 = dist/sqrt(Nf)*ones(Nf, 1);
D = zeros(Nd, Nf);
c = ones(Nd, 1);
%c1 = -c2;

n1 = floor(Nd/2);
sig = r*eye(Nf);
D(1:n1, :) = mvnrnd(c1, sig, n1);

n2 = Nd - n1;
D(n1+1:Nd, :) = mvnrnd(c2, sig, n2);
c(n1+1:Nd) = -1;

nD = randn(Nd, Nnf);

D= [D nD];
D = zscore(D); %zscore, each column~N(0, 1)
end