# gradient descent for logit matrix recovery, implemented with gpu using cupy
# this is much faster than the numpy implementation in gradient_descent.py
# we use this algorithm in our simulations

from helper_fns import *
import cupy
from cupyx.scipy.sparse.linalg import eigsh


# Helper functions

def symmetrize_gpu(A):
    return (A + A.T) / 2

def logit_gpu(p):
    # Small epsilon to avoid log(0) or division by zero
    eps = 1e-12
    p = cupy.clip(p, eps, 1 - eps)
    return cupy.log(p / (1 - p))

def sigmoid_gpu(X):
    # stable sigmoid
    return cupy.where(X >= 0, 1 / (1 + cupy.exp(-X)), cupy.exp(X) / (1 + cupy.exp(X)))

def topk_psd_factor_gpu(G, k):
    """
    Returns Z and its operator norm ||Z||_2.
    """
    G = symmetrize_gpu(G)
    n = G.shape[0]
    if k <= 0:
        return cupy.zeros((n, 0), dtype=G.dtype), 0.0

    # Get top k eigenvalues
    vals, vecs = eigsh(G, k=k, which="LA")
    idx = cupy.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    # The operator norm of Z0 is sqrt(max_eigenvalue of G)
    # We take max(vals[0], 0) because G might have tiny negative eigs due to precision
    op_norm = cupy.sqrt(cupy.maximum(vals[0], 0.0))

    vals_clamped = cupy.maximum(vals, 0.0)
    Z0 = vecs * cupy.sqrt(vals_clamped + 1e-18)

    return Z0, op_norm


# usvt initialization, which returns the initial estimate Z0 
# along with its operator norm, which is used in determining gradient descent step size

def init_usvt_gpu(
    A,              # adjacency matrix of the graph
    d,              # dimension of the latent features
    tau=None,       # singular value threshold. If left to be None, will use the empirical degree calculated from A
    usvt_rank=100,  # how many leading eigs of A to compute for USVT truncation
):
    n = A.shape[0]

    A = symmetrize_gpu(A)
    cupy.fill_diagonal(A, 0.0)

    p = cupy.mean(A)
    if tau is None:
        tau = cupy.sqrt(cupy.maximum(n * p, 1e-12))

    usvt_rank = int(min(max(usvt_rank, d + 5), n - 1))

    # eigsh on GPU
    # 'LA' = Largest Algebraic
    vals, vecs = eigsh(A, k=usvt_rank, which="LA")

    # Sort descending (eigsh usually returns ascending)
    idx = cupy.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    keep = vals >= tau
    if not cupy.any(keep):
        keep[: min(d, len(vals))] = True

    # Pr = (V * vals) @ V.T
    Pr = (vecs[:, keep] * vals[keep]) @ vecs[:, keep].T
    Pr = symmetrize_gpu(Pr)

    # Clipping and Logit
    Pp = cupy.clip(Pr, 0, 1)
    Theta_hat = logit_gpu(Pp)

    Z0, z0_norm = topk_psd_factor_gpu(Theta_hat, d)

    return Z0, z0_norm

# gradient descent step
# the default inputs are the parameters used in our simulations
# loss typically stabilize at ~800 GD steps

def gradient_descent_gpu(A, d, eta=0.2, T=1000, verbose=False, init_kwargs = None):

    init_kwargs = init_kwargs or {}

    # Move to cupy
    A = cupy.asarray(A, dtype=cupy.float64)

    Z_init, Z_init_norm = init_usvt_gpu(A, d, **init_kwargs) # in gpu

    Z = Z_init

    # Step size: eta_Z = eta / ||Z0||_op^2
    op_sq = Z_init_norm ** 2
    eta_Z = eta / max(op_sq, 1e-18)

    history = []
    for t in range(T):
        Theta = Z @ Z.T                   # n×n  
        S = sigmoid_gpu(Theta)            # n×n  

        # Gradient: 2 * (A - S) @ Z
        Z = Z + 2.0 * eta_Z * ((A - S) @ Z)  

        if verbose and (t % 100 == 0 or t == T - 1):
            # Compute loss
            # L = -sum(A*Theta) + sum(log(1+exp(Theta)))
            loss = cupy.sum(cupy.maximum(Theta, 0) - A * Theta + cupy.log1p(cupy.exp(-cupy.abs(Theta))))
            # Convert just the scalar for printing
            loss_val = float(loss)
            history.append(loss_val)
            print(f"t={t} loss={loss_val:.6e}")

    return Z, Z @ Z.T, history
