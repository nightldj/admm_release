%% general admm solver
% author: Zheng Xu, xuzhustc@gmail.com, Apr. 2016
% objective functioin: min h(u)+g(v) st. Au+Bv=b
% input:
%   solvh: function update u from v,l
%   solvg: function update v from Au,l, note that solvg requires Au as
%       input instead of u to use the relaxation term
%   A: linear constraint Au + Bv = b
%   At: the transpose of A
%   B: linear constraint Au + Bv = b
%   b: linear constraint Au + Bv = b
%   v0: the initialization of v
%   l0: the initialization of dual varialbe \lambda
% output:
%   sol: solution to the problem,
%       sol.u, sol.v, sol.l
%   outs:
%       outs.pres  %primal residual
%       outs.dres  %dual residual
%       outs.mres  %monotone residual
%       outs.taus  %penalty parameter
%       outs.gammas %relaxation parameter
%       outs.objs  %objective
%       outs.tols %relative residual
%       outs.iter  %total iteration before convergence

function [sol, outs, opts] = aradmm_core(solvh, solvg, A, At, B, b, v0, l0, opts)

%parameter, parse options
%general
maxiter = opts.maxiter;
tol = opts.tol; %relative tol for stop criterion
minval = 1e-20; %max(opts.tol/10, 1e-20); %smallest value considered
tau = max(opts.tau, minval); %initial stepsize
adp = opts.adp_flag;
verbose = opts.verbose;
obj = opts.obj;
%ARADMM
freq = opts.adp_freq; %adaptive stepsize, update frequency
siter = max(opts.adp_start_iter-1, 1); %start iteration for adaptive stepsize, at least 1, then start at siter+1
eiter = min(opts.adp_end_iter, maxiter)+1; %end iteration for adaptive stepsize, at most the maximum iteration number
orthval = max(opts.orthval, minval);  %value to test correlation, curvature could be estimated or not
%Residual balance
bs = opts.beta_scale; %the scale for updating stepsize, tau = bs * tau or tau/bs, 2 in the paper
rs = opts.res_scale; %the scale for the criterion, pres/dres ~ rs or 1/rs, 0.1 in the paper
%relaxation parameter
gamma = opts.gamma;
gamma0 = 1.5;
gmh = 1.9;
gmg = 1.1;

%record
pres = zeros(maxiter, 1); % primal residual
dres = zeros(maxiter, 1); % dual residual
mres = zeros(maxiter, 1);  % the monotone residual
taus = zeros(maxiter+1, 1); % penalty
objs = zeros(maxiter, 1); % objective
tols = zeros(maxiter, 1); % relative residual
gms = zeros(maxiter+1, 1); % relaxation parameter
taus(1) = tau;
gms(1) = gamma;

%initialize
v1 = v0;
Bv1 = B(v1);
Bv0 = Bv1;
l1 = l0;
b_norm = norm(b(:));

for iter = 1:maxiter
    %update u
    u = solvh(v1, l1, tau);
    Au = A(u);
    
    %update v
    au =  gamma*Au + (1.0-gamma)*(b-Bv1);  %relax
    v = solvg(au, l1, tau);
    Bv = B(v);
    
    %update l
    l = l1 + tau*(b-au-Bv);
    
    %residual
    pres1 = b-Au-Bv;
    pres(iter) = norm(pres1(:));  %primal residual
    dres1 = At(Bv-Bv1);
    dres(iter) = tau*norm(dres1(:)); %dual residual
    mres(iter) = tau*pres(iter)^2+tau*norm(Bv(:)-Bv1(:))^2; %monotone residual
    if verbose > 1
        objs(iter) = obj(u, v); %objective
    end
    
    %stop criterion
    tmp = At(l);
    pres_norm = pres(iter)/max([norm(Au(:)),norm(Bv(:)),b_norm]);
    dres_norm = dres(iter)/norm(tmp(:));
    tols(iter) = max(pres_norm, dres_norm);
    if verbose
        fprintf('%d ADMM iter: %d, tol: %e\n', adp, iter, tols(iter));
    end
    if tols(iter) < tol
        break;
    end
    
    %% adaptive stepsize
    switch adp
        case 1 %AADMM with spectral penalty
            if iter == 1 %record at first iteration
                l0 = l;
                l_hat0 = l1 + tau*(b-Au-Bv1);
                Bv0 = Bv;
                Au0 = Au;
            elseif mod(iter,freq)==0 && iter>siter && iter < eiter   %adaptive stepsize
                %l_hat
                l_hat = l1 + tau*(b-Au-Bv1);
                
                tau = aadmm_estimate(iter, tau, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose);
                
                % record for next estimation
                l0 = l;
                l_hat0 = l_hat;
                Bv0 = Bv;
                Au0 = Au;
            end %frequency if, AADMM
        case 3 %residual balancing
            if iter>siter && iter < eiter
                if dres(iter) < pres(iter) * rs %dual residual is smaller, need large tau
                    tau = bs * tau;
                elseif pres(iter) < dres(iter) * rs %primal residual is smaller, need small tau
                    tau = tau/bs;
                    %else: same tau
                end
            end %converge if, RB
        case 5 %ARADMM
            if iter == 1 %record at first iteration
                l0 = l;
                l_hat0 = l1 + tau*(b-Au-Bv1);
                Bv0 = Bv;
                Au0 = Au;
            elseif mod(iter,freq)==0 && iter>siter && iter < eiter   %adaptive stepsize
                %l_hat
                l_hat = l1 + tau*(b-Au-Bv1);
                
                [tau, gamma] = aradmm_estimate(iter, tau, gamma, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose, gmh, gmg, gamma0);
                
                % record for next estimation
                l0 = l;
                l_hat0 = l_hat;
                Bv0 = Bv;
                Au0 = Au;
            end %frequency if, AADMM
    end %adaptive switch
    %end of adaptivity
    
    
    %%
    taus(iter+1) = tau;
    gms(iter+1) = gamma;
    Bv1 = Bv;
    v1 = v;
    l1 = l;
end
sol.u = u;
sol.v = v;
sol.l = l;
outs.pres = pres(1:iter); %primal residual
outs.dres = dres(1:iter); %dual residual
outs.mres = mres(1:iter); %monotone residual
outs.taus = taus(1:iter); %penalty parameter
outs.gammas = gms(1:iter); %relaxation parameter
outs.objs = objs(1:iter); %objective
outs.tols = tols(1:iter); %relative residual
outs.iter = iter; %total iteration before convergence
end

function tau_h = curv_adaptive_BB(al_h, de_h)
%adapive BB, reference: FASTA paper
tmph = de_h/al_h; %correlation
if tmph > .5
    tau_h = de_h;
else
    tau_h = al_h - 0.5*de_h;
end
end

function [tau, gamma] = aradmm_estimate(iter, tau, gamma, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose, gmh, gmg, gamma0)
%inner product
tmp = real(conj(Au-Au0).*(l_hat-l_hat0));
ul_hat = sum(tmp(:));
tmp = real(conj(Bv-Bv0).*(l-l0));
vl = sum(tmp(:));

%norm of lambda, lambda_hat
tmp = l_hat-l_hat0;
dl_hat = norm(tmp(:));
tmp = l-l0;
dl = norm(tmp(:));

%norm of gradient change
tmp = Au-Au0;
du = norm(tmp(:));
tmp = Bv-Bv0;
dv = norm(tmp(:));


%flag to indicate whether the curvature can be estimated
hflag = false;
gflag = false;

%estimate curvature, only if it can be estimated
%use correlation/othogonal to test whether can be estimated
%use the previous stepsize when curvature cannot be estimated
if ul_hat > orthval*du*dl_hat + minval
    hflag = true;
    al_h = dl_hat^2/ul_hat;
    de_h = ul_hat/du^2;
    bb_h = curv_adaptive_BB(al_h, de_h);
end
if vl > orthval*dv*dl + minval
    gflag = true;
    al_g = dl^2/vl;
    de_g = vl/dv^2;
    bb_g = curv_adaptive_BB(al_g, de_g);
end

if hflag && gflag
    ss_h = sqrt(bb_h);
    ss_g = sqrt(bb_g);
    gamma = min(1 + 2/(ss_h/ss_g+ss_g/ss_h), gamma0);
    tau = ss_h*ss_g;
elseif hflag
    gamma = gmh; %1.9;
    tau = bb_h;
elseif gflag
    gamma = gmg; %1.1;
    tau = bb_g;
else
    gamma = gamma0; %1.5;
    %tau = tau;
end

if verbose == 3
    if ul_hat < 0 || vl < 0
        fprintf('(%d) <u, l>=%f, <v, l>=%f\n', iter, ul_hat, vl);
    end
    if hflag
        fprintf('(%d) corr_h=%f,  al_h=%f,  estimated tau=%f,  gamma=%f \n', iter,...
            ul_hat/du/dl_hat, al_h, tau, gamma);
    end
    if gflag
        fprintf('(%d) corr_g=%f,  al_g=%f,  tau=%f gamma=%f\n', iter,...
            vl/dv/dl, al_g, tau, gamma);
    end
    if ~hflag && ~gflag
        fprintf('(%d) no curvature, corr_h=%f, corr_g=%f, tau=%f,  gamma=%f\n', iter,...
            ul_hat/du/dl_hat, vl/dv/dl, tau, gamma);
    end
end
end

%% AADMM spectral penalty parameter
function tau = aadmm_estimate(iter, tau, Au, Au0, l_hat, l_hat0, Bv, Bv0, l, l0, orthval, minval, verbose)
%inner product
tmp = real(conj(Au-Au0).*(l_hat-l_hat0));
ul_hat = sum(tmp(:));
tmp = real(conj(Bv-Bv0).*(l-l0));
vl = sum(tmp(:));

%norm of lambda, lambda_hat
tmp = l_hat-l_hat0;
dl_hat = norm(tmp(:));
tmp = l-l0;
dl = norm(tmp(:));

%norm of gradient change
tmp = Au-Au0;
du = norm(tmp(:));
tmp = Bv-Bv0;
dv = norm(tmp(:));


%flag to indicate whether the curvature can be estimated
hflag = false;
gflag = false;

%estimate curvature, only if it can be estimated
%use correlation/othogonal to test whether can be estimated
%use the previous stepsize when curvature cannot be estimated
if ul_hat > orthval*du*dl_hat + minval
    hflag = true;
    al_h = dl_hat^2/ul_hat;
    de_h = ul_hat/du^2;
    bb_h = curv_adaptive_BB(al_h, de_h);
end
if vl > orthval*dv*dl + minval
    gflag = true;
    al_g = dl^2/vl;
    de_g = vl/dv^2;
    bb_g = curv_adaptive_BB(al_g, de_g);
end

%if curvature can be estimated for both terms, balance the two
%if one of the curvature cannot be estimated, use the estimated one
%or use the previous stepsize to estimate
if hflag && gflag
    tau = sqrt(bb_h*bb_g);
elseif hflag
    tau = bb_h;
elseif gflag
    tau = bb_g;
    %else
    %tau = tau;
end

if verbose == 3
    fprintf('(%d) <u, l>=%f, <v, l>=%f\n', iter, ul_hat, vl);
    if hflag
        fprintf('(%d) corr_h=%f,  al_h=%f,  estimated tau=%f \n', iter, ...
            ul_hat/du/dl_hat, al_h, tau);
    end
    if gflag
        fprintf('(%d) corr_g=%f,  al_g=%f,  estimated tau=%f \n', iter,...
            vl/dv/dl, al_g, tau);
    end
    if ~hflag && ~gflag
        fprintf('(%d) no curvature, corr_h=%f, corr_g=%f, tau=%f \n', iter,...
            ul_hat/du/dl_hat, vl/dv/dl, tau);
    end
end
end