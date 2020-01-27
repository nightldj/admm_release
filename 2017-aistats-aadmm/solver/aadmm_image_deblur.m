% demo 
% use the AADMM solver
% details in Adaptive ADMM with Spectral Penalty Parameter
% Implementation, AISTATS 2017
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_image_deblur(x_given, fbkernel, mu, lam1, lam2, opts)

%% subproblem solver

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

% mu/2 ||K*u-f||^2
h = @(x) 0.5*mu*norm(ifft2(fbkernel.*fft2(x))-x_given, 'fro')^2;  %regression term
%lam1 |x| + lam2/2 ||x||^2
g = @(x) lam1*norm(x(:),1) + 0.5*lam2*norm(x(:))^2; %regularizer
%objective
obj = @(u, v) h(u)+g(grad2d(u));
% min mu/2 ||u-f||^2 + t/2 ||-A(u) + v + l/t||^2
%  opt condition:  (mu K'K + t A'A)*u = mu*F'K'Ff + A'(t*v+l)
rhsh = @(v, l, t) mu*ifft2(conj(fbkernel).*fft2(x_given))+grad2d_conj(t*v+l);
fs2 = conj(fsx).*fsx + conj(fsy).*fsy;
fk2 = conj(fbkernel).*fbkernel;
solvh = @(v, l, t) real( ifft2(  1.0./max((mu*fk2+t*fs2), 1e-20) .* fft2(rhsh(v,l,t)) ) );
% min lam1 |x| + lam2/2 |x|^2 + 1/(2t)|| x - z ||^2 
%  opt condition:  shrink(z, lam1*t)/(1+lam2*t)
shrink = @(x,t) sign(x).*max(abs(x) - t,0);
proxg = @(z,t) shrink(z, lam1*t)/(1+lam2*t);
solvg = @(u, l, t) proxg(grad2d(u)-l/t, 1/t); %update v

opts.obj = obj; %objective function, used when verbose


%% initialization
x0 = grad2d(x_given);
l0 = ones(size(x0));

tic;
[sol, outs] = aadmm_core(solvh, solvg, grad2d, grad2d_conj, @(x)-x, 0, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;
end