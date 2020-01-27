# ADMM optimization 

This repository includes Matlab and/or Python implementation of (adaptive) ADMM optimization for various applications in a series of my previous works that make part of my thesis [''Alternating Optimization: Constrained Problems, Adversarial Networks, and Robust Models''](https://drum.lib.umd.edu/handle/1903/25045).  

The codes for adaptive relaxed ([ARADMM, CVPR'17](http://openaccess.thecvf.com/content_cvpr_2017/html/Xu_Adaptive_Relaxed_ADMM_CVPR_2017_paper.html)), adaptive consensus ADMM ([ACADMM, ICML'17](http://proceedings.mlr.press/v70/xu17c.html)) and low-rank least squares for visual subcategories ([BMVC'15](http://www.bmva.org/bmvc/2015/papers/paper149/paper149.pdf)) have been previously released on my [personal webpage](https://sites.google.com/site/xuzhustc/resume). The codes for adaptive ADMM ([AADMM, AISTATS'17](https://arxiv.org/abs/1605.07246)), AADMM for nonconvex problems ([NeurIPS workshop'16](https://arxiv.org/abs/1612.03349)), and adaptive multi-block ADMM ([my thesis, chapter 5.1](https://drum.lib.umd.edu/handle/1903/25045)) are included in this package. We also provide implementation for baseline methods, vanilla ADMM, [Fast (Nestrov) ADMM](https://epubs.siam.org/doi/abs/10.1137/120896219), [residual balancing](https://link.springer.com/article/10.1023/A:1004603514434), and normalized residual balancing ([my thesis, chapter 5.1](https://drum.lib.umd.edu/handle/1903/25045)). 

## Applications
We provide ADMM-based solvers for various applications, include 
+ linear regression with elastic net (l2 + l1) regularizer
+ linear regression with sparse (l1/l0) regularizer
+ logistic regression with (l1/l2) regualarizer
+ basis pursuit
+ low-rank least squares
+ robust PCA (RPCA)
+ quadratic programming (QP)
+ semidefinite programming (SDP)
+ support vector machines (SVMs)
+ 1D/2D denoising with total variation regualrizer
+ image denoising/restoration/deblurring with total variation regularizer
+ distributed consensus problem: logistic regression
+ distributed consensus problem: linear regression
+ exemplar nonconvex problems: eigenvalue problem
+ exemplar nonconvex problems: phase retrieval

You might also be interested to check our repository for applying ADMM to neural networks ([ICML'16](https://gitlab.umiacs.umd.edu/tomg/admm_nets)), and tensor ([NeurIPS workshop'16](https://github.com/nightldj/tensor_notf))


## Convergence rate
We show under mild conditions, adaptive ADMM methods are guaranteed to converge with O(1/k) rate in [ACADMM, ICML'17](http://proceedings.mlr.press/v70/xu17c.html)



## Citation
If you find our implementation helpful, please kindly consider cite our papers. 

For general usage of the package, refer to either the first AADMM paper (AISTATS'17) or the PhD thesis,
```
@article{xu2016adaptive,
	Author = {Xu, Zheng and Figueiredo, Mario AT and Goldstein, Tom},
	Journal = {AISTATS},
	Title = {Adaptive {ADMM} with Spectral Penalty  Parameter Selection},
	Year = {2017}
}

@phdthesis{xu2019alternating,
  title={Alternating Optimization: Constrained Problems, Adversarial Networks, and Robust Models},
  author={Xu, Zheng},
  year={2019}
}
```

For multi-block ADMM, and baseline method normalized residual balancing, refer to the PhD thesis,
```
@phdthesis{xu2019alternating,
  title={Alternating Optimization: Constrained Problems, Adversarial Networks, and Robust Models},
  author={Xu, Zheng},
  year={2019}
}
```

For relaxed ADMM, and related applications, refer to the ARADMM paper (CVPR'17), 
```
@article{xu2017adaptive,
	Author = {Xu, Zheng and Figueiredo, Mario AT and Yuan, Xiaoming and Studer, Christoph and Goldstein, Tom},
	Journal = {CVPR},
	Title = {Adaptive Relaxed {ADMM}: Convergence Theory and Practical Implementation},
	Year = {2017}
}
```

For consensus ADMM, and related applications, refer to the ACADMM paper (ICML'17),
```
@article{xu2017acadmm,
	Author = {Xu, Zheng and Taylor, Gavin and Li, Hao and Figueiredo, Mario AT and Yuan, Xiaoming and Goldstein, Tom},
	Journal = {ICML},
	Title = {Adaptive Consensus {ADMM} for Distributed Optimization},
	Year = {2017}
}
```

For nonconvex applications, refer to the workshop paper
```
@inproceedings{xu2016empirical,
	Author = {Xu, Zheng and De, Soham and Figueiredo, Mario A. T. and Studer, Christoph and Goldstein, Tom},
	Booktitle = {NIPS workshop on nonconvex optimization},
	Title = {An Empirical Study of {ADMM} for Nonconvex Problems},
	Year = {2016}
}
```

For the subcategorizatoin application, refer to 
```
@inproceedings{xu2015exploiting,
	Author = {Zheng Xu and Xue Li and Kuiyuan Yang and Tom Goldstein},
	Booktitle = {BMVC, Swansea, UK, September 7-10, 2015},
	Title = {Exploiting Low-rank Structure for Discriminative Sub-categorization},
	Year = {2015}
}
```
