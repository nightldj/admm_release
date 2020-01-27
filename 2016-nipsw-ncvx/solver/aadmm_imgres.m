% demo 
% use the AADMM solver
% details in An Empirical Study of ADMM for Nonconvex Problems
% @author: Zheng Xu, xuzhustc@gmail.com

function [sol, outs] = aadmm_imgres(x_given, mu, lam1, opts)


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
%lam1 |x| + lam2/2 ||x||^2
g = @(x) lam1*sum(abs(x(:))>1e-20); %regularizer
%objective
obj = @(u, v) h(u)+g(grad2d(u));
% min mu/2 ||u-f||^2 + t/2 ||-A(u) + v + l/t||^2
%  opt condition:  (mu + t A'A)*u = mu*f + A'(t*v+l)
rhsh = @(v, l, t) mu*x_given+grad2d_conj(t*v+l);
fs2 = conj(fsx).*fsx + conj(fsy).*fsy;
solvh = @(v, l, t) real( ifft2(  1./(mu+t*fs2) .* fft2(rhsh(v,l,t)) ) );
% min lam1 |x| + lam2/2 |x|^2 + 1/(2t)|| x - z ||^2 
%  opt condition:  
proxg = @(z,t) ( (z.^2-2*t) >0 ).*z;
solvg = @(u, l, t) proxg(grad2d(u)-l/t, lam1/t); %update v

fA = @(x) grad2d(x);
fAt = @(x) grad2d_conj(x);
fB = @(x) -x;
fb = 0;


opts.obj = obj; %objective function, used when verbose

%% initialization
x0 = grad2d(x_given);
l0 = ones(size(x0));


tic;
[sol, outs] = aadmm_core(solvh, solvg, grad2d, grad2d_conj, @(x)-x, 0, x0, l0, opts);
outs.runtime  = toc;
sol = sol.u;

end
