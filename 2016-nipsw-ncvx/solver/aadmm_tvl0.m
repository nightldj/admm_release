% demo 
% use the AADMM solver
% details in An Empirical Study of ADMM for Nonconvex Problems
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_tvl0(x_given, mu, lam1, opts)

%% minimize  mu/2 ||x-f||^2 + lam1 |\grad x|  
% stencil for gradient
sx = zeros(size(x_given));
sx(1)=-1;
sx(end)=1;
fsx = fft(sx);
%gradient operator
grad1d = @(X) ifft(fsx.*fft(X));
grad1d_conj = @(G) real(ifft( conj(fsx).*fft(G)) );

% mu/2 ||u-f||^2
h = @(x) 0.5*mu*norm(x-x_given)^2;  %regression term
%lam1 |x|_0
g = @(x) lam1*sum(abs(x)>1e-20); %regularizer
%objective
obj = @(u, v) h(u)+g(grad1d(u));
% min mu/2 ||u-f||^2 + t/2 ||-A(u) + v + l/t||^2
%  opt condition:  (mu + t A'A)*u = mu*f + A'(t*v+l)
rhsh = @(v, l, t) mu*x_given+grad1d_conj(t*v+l);
fs2 = conj(fsx).*fsx;
solvh = @(v, l, t) real( ifft(  1./(mu+t*fs2) .* fft(rhsh(v,l,t)) ) );
% min lam1 |x|_0 + 1/(2t)|| x - z ||^2 
proxg = @(z,t) ( (z.^2-2*t) >0 ).*z;
solvg = @(u, l, t) proxg(grad1d(u)-l/t, lam1/t); %update v
fA = @(x) grad1d(x);
fAt = @(x) grad1d_conj(x);
fB = @(x) -x;
fb = 0;

opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = grad1d(x_given);
l0 = zeros(size(x0));

tic;
[sol, outs] = aadmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;

end
