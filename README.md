# Semi-supervised learning on latent space graphs
Code for numerical simulations with synthetic data. 

Algorithm 2 (Recover) is in `convex_optimization.py`  
Algorithm 3, 4 (Regress, EigFix) in `helper_fns.py`  
Algorithms 5, 6 (USVTInitialization, GradientDescent) in `gradient_descent.py` (using numpy), and identically in `gradient_descent_gpu.py` (faster cupy implementation)
Newton's method implementation of Laplacian smoothing / regularization in `laplacian_smoothing.py`
