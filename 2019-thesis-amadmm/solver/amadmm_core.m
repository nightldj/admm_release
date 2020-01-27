%% general multiblock admm method
% author: Zheng Xu, xuzhustc@gmail.com, Apr. 2016
% objective functioin: min \sum h(u) st. \sum Au =b
% input:
%   solvh: function update u 
%   A:
%   At:
%   b:
%   u0:
%   l0:
% output:

function [sol, outs, opts] = amadmm_core(solvh, A, At, b, u0, l0, opts)

%parameter, parse options
%general
maxiter = opts.maxiter;
tol = opts.tol; %relative tol for stop criterion
minval = 1e-20; %max(opts.tol/10, 1e-20); %smallest value considered
tau = max(opts.tau, minval); %initial stepsize
adp = opts.adp_flag;
verbose = opts.verbose;
obj = opts.obj;
%AADMM
freq = opts.adp_freq; %adaptive stepsize, update frequency
siter = max(opts.adp_start_iter-1, 1); %start iteration for adaptive stepsize, at least 1, then start at siter+1
eiter = min(opts.adp_end_iter, maxiter)+1; %end iteration for adaptive stepsize, at most the maximum iteration number
orthval = max(opts.orthval, minval);  %value to test orhgonal/correlation, curvature could be estimated or not
%Residual balance
bs = opts.beta_scale; %the scale for updating stepsize, tau = bs * tau or tau/bs, 2 in the paper
rs = opts.res_scale; %the scale for the criterion, pres/dres ~ rs or 1/rs, 0.1 in the paper
%optimize options for stepsize
fmin_opts = optimoptions(@fminunc,'Algorithm','quasi-newton', 'GradObj', 'on', 'MaxIter', 200, 'TolFun', 1e-3, 'TolX', 1e-3, 'Display','off');


%record
pres = zeros(maxiter, 1);
dres = zeros(maxiter, 1);
taus = zeros(maxiter+1, 1);
objs = zeros(maxiter, 1);
tols = zeros(maxiter, 1);
taus(1) = tau;


%initialize
N = length(solvh); %block number
u = u0; %all u
u1 = u; %u from previous step
Au = zeros(length(l0), N); %Au
for i=1:N
    Au(:, N) = A{i}(u{i});
end
Au1 = Au; %Au from previous step
csAu = cumsum(Au, 2); %cum sum of Au
csAu1 = csAu; %cum sum from previous step
l = repmat(l0, 1, N); %dual variables
l1 = l; %dual variable from previous step
b_norm = norm(b(:));
dres_vec = zeros(N-1, 1); %dual residual for multi-block
spectral = zeros(N, 1);
for iter = 1:maxiter
    %update u
    for bi=1:N
        u{bi} = solvh{bi}(u, l1(:, N), tau);
        Au(:, bi) = A{bi}(u{bi});
    end
    
    %update l
    pres1 = b-sum(Au, 2);
    l(:, N) = l1(:, N) + tau*pres1;
    
    %residual
    pres(iter) = norm(pres1(:));  %primal residual
    csAu = cumsum(Au, 2);
    csResAu = csAu - csAu1;
    csResAu = csResAu(:, 2:end) - repmat(csResAu(:, 1), 1, N-1);
    for i=1:N-1
        dres_vec(i) = norm(At{i}(csResAu(:, i)));
    end
    dres(iter) = max(tau*dres_vec); %max/sum, dual residual
    if verbose
        fprintf('%d ADMM iter: %d\n', adp, iter);
    end
    if verbose > 1
        objs(iter) = obj(u); %objective
    end
    
    %stop criterion
    pres_norm = pres(iter)/max([max(sqrt(sum(Au.*Au, 1))) b_norm minval]);
    for i=1:N-1
        dres_vec(i) = norm(At{i}(l(:, N)));
    end
    dres_vec = dres_vec(dres_vec > minval);
    dres_norm = dres(iter)/min(dres_vec);
    tols(iter) = max(pres_norm, dres_norm);
    if tols(iter) < tol
        break;
    end
    
    %% adaptive stepsize
    switch adp
        case {1, 5} %AADMM with spectral penalty
            if iter == 1 %record at first iteration
                for bi=1:N-1
                    l(:, bi) = l1(:, N) + tau*(b - csAu(:, bi) - csAu1(:, N) + csAu1(:, bi)); %intermediate l
                end
                l0 = l;
                Au0 = Au;
            elseif mod(iter,freq)==0 && iter>siter && iter < eiter   %adaptive stepsize
                for bi=1:N-1
                    l(:, bi) = l1(:, N) + tau*(b - csAu(:, bi) - csAu1(:, N) + csAu1(:, bi)); %intermediate l
                end
                
                tmpAu = Au-Au0; %Au: gradient change
                tmpl = l-l0; %lambda: variable change
                ul = sum(tmpAu.*tmpl, 1); %inner product
                dl = sqrt(sum(tmpl.*tmpl, 1)); %norm of lambda change
                du = sqrt(sum(tmpAu.*tmpAu, 1)); %norm of Au change
                
                %safeguard & curvature estimate
                cvflag = (ul > (orthval.*du.*dl + minval));
                if sum(cvflag) > 0 %at least one cruvature can be estimated
                    cv_al = dl(cvflag).^2./ul(cvflag);
                    cv_de = ul(cvflag)./du(cvflag).^2;
                    bb_h = curv_adaptive_BB(cv_al, cv_de);
                                       
                    %use other curves to estimate the difficult curves
                    spectral(cvflag) = bb_h;
                    spectral(~cvflag) = max(bb_h); %min(bb_h) % max(bb_h), good for EN % median(bb_h); %when curvature canot be estimated
                    if 1 == adp
                        if tau < 0.1 || tau > 1000
                            fmin_opts.optTol = min(tau*0.01, 0.01/tau);
                        end
                        opt_tau = fminunc(@(x) optm_tau(x, spectral), tau, fmin_opts);
                    elseif 5 == adp
                        %  Use the geometric mean and a hueristic instead of
                        %  true minimizer
                        opt_tau = geomean(spectral);
                    end
                    if opt_tau > 0
                        tau  = opt_tau;
                    else
                        fprintf('(%d) MB_AADMM: negative tau %.3f, previous tau %.3f\n', iter, opt_tau, tau);
                    end
                    
                    if verbose >2
                        fprintf('(%d) tau: %.3f, curve flag: %s, \n', iter,...
                            tau, mat2str(uint8(cvflag)));
                        fprintf('(%d) tau: %.3f, split stepsizes: %s, \n', iter,...
                            tau, mat2str(bb_h));
                    end
                end
                
                % record for next estimation
                l0 = l;
                Au0 = Au;
            end %frequency if, AADMM
        case 2 %Fast ADMM, Nesterov with restart
            if 1 == iter
                fprintf('(%d) Nesterov ADMM not supproted yet', iter);
            end
        case 3 %residual balancing
            if iter>siter && iter < eiter
                if dres(iter) < pres(iter) * rs %dual residual is smaller, need large tau
                    tau = bs * tau;
                elseif pres(iter) < dres(iter) * rs %primal residual is smaller, need small tau
                    tau = tau/bs;
                    %else: same tau
                end
            end %converge if, RB
        case 4 %normalized residual balancing
            if iter>siter && iter < eiter
                if dres_norm < pres_norm * rs %dual residual is smaller, need large tau
                    tau = bs * tau;
                elseif pres_norm < dres_norm * rs %primal residual is smaller, need small tau
                    tau = tau/bs;
                    %else: same tau
                end
            end %converge if, NRB
    end %adaptive switch
    %end of adaptivity
    
    
    %%
    taus(iter+1) = tau;
    if adp~=2
        Au1 = Au;
        csAu1 = csAu;
        u1 = u;
        l1 = l;
    end
end
sol.u = u;
outs.pres = pres(1:iter);
outs.dres = dres(1:iter);
outs.taus = taus(1:iter);
outs.objs = objs(1:iter);
outs.tols = tols(1:iter);
outs.iter = iter;
end

function tau_h = curv_adaptive_BB(al_h, de_h)
%adapive BB, reference: FASTA paper of Tom
tmph = de_h./al_h; %correlation
flag = tmph > .5;
tau_h = zeros(size(al_h));
tau_h(flag) = de_h(flag);
tau_h(~flag) = al_h(~flag) - 0.5*de_h(~flag);
end

function [f, g] = optm_tau(t, spectral)
tmp = t+spectral;
f = sum(log(tmp)) - log(t);
g = sum(1./tmp) - 1/t;
end