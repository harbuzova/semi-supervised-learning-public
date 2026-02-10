import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix


def newton_lp_laplacian_solver(W, y_labeled, labeled_idx, p=4, max_iter=50, conv_tol=1e-5, cg_tol=1e-6, eps=1e-8, u_init=None):
    """
    Newton's method for minimizing variational p-Laplacian energy with full vectorization.

    Parameters:
        W           : (n x n) sparse adjacency matrix of the graph (can be dense)
        y_labeled   : (len(labeled_idx),) array of known labels
        labeled_idx : indices of labeled nodes
        p           : power in the p-Laplacian (e.g., p = d + 1)
        max_iter    : max number of Newton iterations
        conv_tol    : convergence tolerance on update norm
        cg_tol      : tolerance for CG linear solve
        eps         : small smoothing for numerical stability
    Returns:
        u           : (n,) vector of predicted values (label function)
    """
    n = W.shape[0]                      # Number of nodes
    u = np.zeros(n) if u_init is None else u_init.copy()
    u[labeled_idx] = y_labeled          # Set known labels

    # Identify unlabeled nodes for convergence check
    unlabeled_idx = np.setdiff1d(np.arange(n), labeled_idx)

    # Convert W to COO format to extract edges and weights
    W = W.tocoo()
    I, J, weights = W.row, W.col, W.data   # Edges: I[i] -- J[i] with weight weights[i]
    E = np.vstack((I, J)).T               # (num_edges x 2) edge list

    for k in range(max_iter):
        u_prev = u.copy()                 # Store previous iterate for convergence check

        # Compute differences u_i - u_j for all edges
        ui = u[E[:, 0]]
        uj = u[E[:, 1]]
        diffs = ui - uj

        # Stabilized norm: (u_i - u_j)^2 + eps
        norm_sq = diffs**2 + eps
        norm = norm_sq ** ((p - 2) / 2)   # |u_i - u_j|^{p - 2}

        # Gradient contribution per edge
        grad_contrib = weights * norm * diffs

        # Initialize gradient vector
        grad = np.zeros(n)
        # Accumulate gradient from edges: grad[i] += grad_contrib, grad[j] -= grad_contrib
        np.add.at(grad, E[:, 0], grad_contrib)
        np.subtract.at(grad, E[:, 1], grad_contrib)

        # Hessian diagonal values for each edge
        hess_vals = weights * ((p - 2) * diffs**2 / norm_sq + 1) * norm

        # Build sparse Laplacian-style Hessian matrix in COO format
        # For each edge (i, j), contribute:
        #   H[i, i] += hess_val
        #   H[j, j] += hess_val
        #   H[i, j] -= hess_val
        #   H[j, i] -= hess_val
        all_I = np.concatenate([I, J, I, J])
        all_J = np.concatenate([I, J, J, I])
        all_data = np.concatenate([hess_vals, hess_vals, -hess_vals, -hess_vals])
        H = sp.coo_matrix((all_data, (all_I, all_J)), shape=(n, n)).tolil()

        # Add small regularization to diagonals for numerical stability
        H.setdiag(H.diagonal() + 1e-6)

        # Clamp known labels by setting rows/cols of labeled nodes to identity
        for idx in labeled_idx:
            H[idx, :] = 0
            H[idx, idx] = 1
            grad[idx] = 0

        # Convert to CSR format for CG solver
        H_csr = H.tocsr()

        # Solve linear system H delta = -grad using Conjugate Gradient
        x0 = np.zeros_like(grad)  # Initial guess same shape as grad (n,)
        delta, info = cg(H_csr, -grad, x0=np.zeros_like(grad), rtol=cg_tol, maxiter=max_iter)
        if info != 0:
            print(f"[Newton] Warning: CG did not converge fully (info={info})")

        # Update label vector
        u += delta

        # Check convergence only on unlabeled nodes
        if np.linalg.norm(delta[unlabeled_idx]) < conv_tol:
            print(f"[Newton] Converged at iteration {k}")
            break

    return u


def homotopy_lp_solver(W, y_labeled, labeled_idx, p_target, p_schedule=None,
                       max_iter=50, tol=1e-5, cg_tol=1e-6, eps=1e-8, verbose=True):
    """
    Homotopy method for solving the variational p-Laplacian by gradually increasing p.

    Parameters:
        W            : (n x n) sparse adjacency matrix
        y_labeled    : observed labels
        labeled_idx  : indices of labeled nodes
        p_target     : final target p (e.g., 17)
        p_schedule   : optional list of increasing p values ending in p_target
        max_iter     : Newton max iterations
        tol          : Newton stopping tolerance
        cg_tol       : CG solver tolerance
        eps          : smoothing for numerical stability
        verbose      : if True, print progress info
    Returns:
        u            : (n,) label vector at final p
    """
    n = W.shape[0]
    u = np.zeros(n)
    u[labeled_idx] = y_labeled

    # Default exponential p schedule
    if p_schedule is None:
        p_schedule = [2]
        while p_schedule[-1] < p_target:
            next_p = min(p_target, int(p_schedule[-1] * 2))
            if next_p == p_schedule[-1]: break
            p_schedule.append(next_p)

    if p_schedule[-1] != p_target:
        p_schedule.append(p_target)

    if verbose:
        print(f"[Homotopy] p-schedule: {p_schedule}")

    for p in p_schedule:
        if verbose:
            print(f"\n[Homotopy] Solving for p = {p}")
        u = newton_lp_laplacian_solver(W, y_labeled, labeled_idx, p=p,
                                       max_iter=max_iter, conv_tol=tol,
                                       cg_tol=cg_tol, eps=eps, u_init=u)

    return u
