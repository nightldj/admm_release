MATLAB Code for 

************************************************************

Exploiting Low-Rank Structure for Discriminative Sub-categorization
In BMVC 2015

Contact: Zheng Xu (xuzhustc@gmail.com)

*************************************************************

v1, Sep, 2015

The package includes implementation of LRLSE-LDAs in the paper. Two optimizatin solvers are investigated. The ADMM solver in the paper. And the FASTA solver which uses the forward backward splitting method. We include the FASTA solver since we finds it is sometimes faster when applying to high dimensional data.

Two demos are included in the package. 
demo_classification_set2.m
demo_clustering_GasSenseor.m
The two demos are for clustering and classification, respectively.




Bundled Code
The code is tested under MATLAB R2013b, Windows 7. The code is self-contained. We use codes downloaded from 
http://www.mathworks.com/matlabcentral/fileexchange/23576-min-max-selection
http://www.cs.umd.edu/~tomg/FASTA.html
Please check those sites for that part of the code. Our experiment uses vlfeat, but the released code does not reply on it. You may also consider installing vlfeat from http://www.vlfeat.org/.
The aligment code for the evaluation of clustering result is kindly shared by Shijie Xiao from NTU.
We thank their generous sharing. 


Please cite our paper if you would like to use the code:
Exploiting Low-Rank Structure for Discriminative Sub-categorization
Zheng Xu, Xue Li, Kuiyuan Yang, Tom Goldstein
British Machine Vision Conference (BMVC), 2015

@inproceedings{xu_BMVC2015,
  author    = {Zheng Xu and Xue Li and Kuiyuan Yang and Tom Goldstein},
  title     = {Exploiting Low-Rank Structure for Discriminative Sub-categorization},
  booktitle = {BMVC},
  year      = {2015}
}


Thanks for your interest! Enjoy!