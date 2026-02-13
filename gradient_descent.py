from helper_fns import *
from scipy.sparse.linalg import eigsh, LinearOperator

# GRADIENT DESCENT ON THE RESCALED FEATURE MATRIX
# we use this method to recover the logit matrix and generate the results in our paper

# ----------------------------
# Numerics
# ----------------------------

def logit(p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def symmetrize(A):
    return 0.5 * (A + A.T)


def center_columns(Z):
    return Z - Z.mean(axis=0, keepdims=True)


def apply_J_left_right(M):
    """
    Center the matrix M, so each row and column sums to zero
    Remove row/col means and adds back grand mean.
    """
    row_mean = M.mean(axis=1, keepdims=True)
    col_mean = M.mean(axis=0, keepdims=True)
    grand_mean = M.mean()
    return M - row_mean - col_mean + grand_mean


def power_spectral_norm_sq(Z, n_iter=30, random_state=0):
    """Estimate ||Z||_op^2 via power iteration on Z^T Z."""
    rng = np.random.default_rng(random_state)
    k = Z.shape[1]
    v = rng.normal(size=(k,))
    v /= (np.linalg.norm(v) + 1e-18)
    # iterate v <- (Z^T Z) v / ||.||
    for _ in range(n_iter):
        v = Z.T @ (Z @ v)
        v_norm = np.linalg.norm(v) + 1e-18
        v /= v_norm
    # Rayleigh quotient gives approx largest eigenvalue of Z^T Z = ||Z||_op^2
    Zv = Z @ v
    return float(Zv @ Zv)


def topk_psd_factor(G, k):
    """
    Return Z such that Z Z^T approximates G using top-k eigenpairs.
    Uses eigsh for speed/memory at n~2000+.
    """
    G = symmetrize(G)
    n = G.shape[0]
    if k <= 0:
        return np.zeros((n, 0), dtype=G.dtype)

    # largest algebraic eigenvalues (LA)
    vals, vecs = eigsh(G, k=k, which="LA")
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    vals = np.maximum(vals, 0.0)
    return vecs * np.sqrt(vals + 1e-18)



# ----------------------------
# Initialization: USVT
# Avoid full SVD: since A is symmetric, singular values are |eigs|.
# USVT keeps the top components of A with eig >= tau.
# Allows much faster GD convergence then random initialization
# ----------------------------
def init_usvt(
    A,                # adjacency matrix (dense)
    d,                # latent dimension
    tau=None,         # we use the empirical mean degree in practice
    center=False,      # center the approximated logit matrix in initialization, not needed in practice
    usvt_rank=100,    # how many leading eigs to compute for USVT truncation
):
    """
    USVT initializer without full SVD:
      - compute top usvt_rank eigenpairs of A
      - keep eigenvalues >= tau
      - reconstruct Pr from kept components
      - clip to [0, 1], logit -> Theta_hat
      - R = J Theta_hat J, then Z0 from top-k eig factor.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    A = symmetrize(A)
    np.fill_diagonal(A, 0.0)

    p = A.mean()
    if tau is None:
        tau = np.sqrt(max(n * p, 1e-12)) # the empirical mean degree, which is the tau threshold we use

    # Top eigenpairs of A.
    usvt_rank = int(min(max(usvt_rank, d + 5), n - 1))
    vals, vecs = eigsh(A, k=usvt_rank, which="LA")
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    keep = vals >= tau
    if not np.any(keep):
        # fallback: keep the largest few
        keep[: min(d, len(vals))] = True

    Pr = (vecs[:, keep] * vals[keep]) @ vecs[:, keep].T
    Pr = symmetrize(Pr)

    lo = 0
    hi = 1
    Pp = np.clip(Pr, lo, hi)
    Theta_hat = logit(Pp)

    R = apply_J_left_right(Theta_hat) if center else Theta_hat

    Z0 = topk_psd_factor(R, d)
    Z0 = center_columns(Z0) if center else Z0
    return Z0


# ----------------------------
# Main Algorithm: Projected GD on Z
# where Z satisfies Z = alpha^(1/2)d^(-1/4)X, ZZ^T = Theta
# returns Z and ZZ^T which is the estimated logit matrix
# ----------------------------

def gradient_descent(
    A,                  # adjacency matrix
    d,                  # latent dimension, assumed to be known
    T=200,              # number of gradient descent steps
    eta=0.2,            # unscaled learning rate
    init_kwargs=None,   # for USVT initialization
    verbose=False,
    random_state=0,
):
    
    init_kwargs = init_kwargs or {}

    Z = init_usvt(A, d, **init_kwargs)
    
    # Step size: eta_Z = eta / ||Z0||_op^2 (estimated quickly)
    op_sq = power_spectral_norm_sq(Z, n_iter=30, random_state=random_state)
    eta_Z = eta / max(op_sq, 1e-18)

    history = {"eta_Z": eta_Z, "loss": []}

    for t in range(T):
        Theta = Z @ Z.T
        S = sigmoid(Theta)

        # Z <- Z + 2 eta_Z (A - sigmoid(Theta)) Z
        Z = Z + 2.0 * eta_Z * ((A - S) @ Z)

        if verbose and (t % max(1, T // 10) == 0 or t == T - 1):
            Theta_clip = np.clip(Theta, -50.0, 50.0)
            loss = np.sum(np.log1p(np.exp(Theta_clip)) - A * Theta_clip)
            history["loss"].append(float(loss))
            print(f"iter {t:4d} | loss {loss:.4e}")

    return Z, Z @ Z.T, history
