function opts = get_default_opts(opts)
opts.adp_flag = 5; % default method: ARADMM
opts.adp_freq = 2; %frequency for adaptive penalty
opts.adp_start_iter = 2; %start iteration of adaptive penalty
opts.adp_end_iter = 1000; %end iteration of adaptive penalty
opts.orthval = 0.2; %threshold for correlation validation in ARADMM
opts.beta_scale = 2; %residual balancing parameter
opts.res_scale = 0.1; %residual balancing parameter
opts.gamma = 1; %relaxation parameter
end