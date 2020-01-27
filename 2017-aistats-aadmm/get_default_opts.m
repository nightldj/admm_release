function opts = get_default_opts()
opts.adp_freq = 2; %frequency for adaptive step size
opts.adp_start_iter = 2; %start iteration of adaptive stepsize
opts.adp_end_iter = 1000; %end iteration of adaptive stepsize
opts.orthval = 0.2; %value to test orthogonal or not
opts.beta_scale = 2; %RB
opts.res_scale = 0.1; %RB

opts.maxiter = 300; %max iteration
opts.tau = .1; %initial stepsize
opts.tol = 1e-5; %relevant stop criterion
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration
%3: more print out for debugging adaptive relaxed ADMM
opts.verbose = 0; 
end