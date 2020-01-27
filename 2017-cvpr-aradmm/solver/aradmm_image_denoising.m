function [sol, outs] = aradmm_image_denoising(x_given, mu, lam1, opts)

% demo total vational image denoising
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% minimize  mu/2 ||x-f||^2 + lam1 |\grad x| 


% stencil for gradient
sx = zeros(size(x_given));
sy = zeros(size(sx));
sx(1,1)=-1;
sx(1,end)=1;
fsx = fft2(sx);
sy(1,1)=-1;
sy(end,1)=1;
fsy = fft2(sy);
%gradient operator
grad2d = @(X) cat(3, ifft2(fsx.*fft2(X)), ifft2(fsy.*fft2(X)));
grad2d_conj = @(G) real(ifft2( conj(fsx).*fft2(G(:,:,1)) + conj(fsy).*fft2(G(:,:,2)) ));

% mu/2 ||u-f||^2
h = @(x) 0.5*mu*norm(x(:)-x_given(:))^2;  %regression term
%lam1 |x| 
g = @(x) lam1*norm(x(:),1); %regularizer
%objective
obj = @(u, v) h(u)+g(grad2d(u));
% min mu/2 ||u-f||^2 + t/2 ||-A(u) + v + l/t||^2
%  opt condition:  (mu + t A'A)*u = mu*f + A'(t*v+l)
rhsh = @(v, l, t) mu*x_given+grad2d_conj(t*v+l);
fs2 = conj(fsx).*fsx + conj(fsy).*fsy;
solvh = @(v, l, t) real( ifft2(  1./(mu+t*fs2) .* fft2(rhsh(v,l,t)) ) ); %update u
% min lam1 |x| + 1/(2t)|| x - z ||^2 
%  opt condition:  shrink(z, lam1*t)
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
proxg = @(z,t) shrink(z, lam1*t);
solvg = @(au, l, t) proxg(au-l/t, 1/t); %update v

fA = grad2d;
fAt = grad2d_conj;
fB =  @(x)-x;
fb = 0;

opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = grad2d(x_given);
l0 = ones(size(x0));

%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
