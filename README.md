Online Robust PCA
=================

Batch and Online Robust PCA (Robust Principal Component Analysis) implementation and examples (matlab version).

Robust PCA based on Principal Component Pursuit (**RPCA-PCP**) is the most popular RPCA algorithm which decomposes the observed matrix M into a low-rank matrix L and a sparse matrix S by solving Principal Component Pursuit:

> \min ||L||_* + \lambda ||S||_1

> s.t. L + S = M

where ||.||_* is a nuclear norm, ||.||_1 is L1-norm. 

Please see the [paper](https://ieeexplore.ieee.org/abstract/document/8736886)[[arxiv version](https://arxiv.org/abs/1702.05698)] for details.

### What is inside?
Folder **omwRPCA** contains various batch and online Robust PCA algorithms.

  * pcp.m: Robust PCA based on Principal Component Pursuit (RPCA-PCP). Reference: Candes, Emmanuel J., et al. "Robust principal component analysis." Journal of the ACM (JACM) 58.3 (2011): 11.

  * omwrpca.m: Online Moving Window Robust PCA.

  * omwrpca_cp.m: Online Moving Window Robust PCA with Change Point Detection. A novel online robust principal component analysis algorithm which can track both slowly changing and abruptly changed subspace. The algorithm is also able to automatically discover change points of the underlying low-rank subspace.

example.m: a working example of omwrpca-cp algorithm based on Lobby dataset

### Citation
If you use this package in any way, please cite the following preprint.
```
@article{xiao2019onlineRPCA,
  title={Online Robust Principal Component Analysis with Change Point Detection},
  author={W. {Xiao} and X. {Huang} and F. {He} and J. {Silva} and S. {Emrani} and A. {Chaudhuri}},
  journal={IEEE Transactions on Multimedia},
  year={2019}
}
```

### Authors
He Fan, Wei Xiao, Xiaolin Huang       

### Contacts
Wei Xiao (<wxiao0421@gmail.com>) 

