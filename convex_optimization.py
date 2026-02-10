from helper_fns import *
import cvxpy as cp

########### CONVEX OPTIMIZATION ###########

def convex_opt(A, lam = None, c_off_diag = None, c_diag = None, verbose=False, max_iters=5000, eps=1e-6):
    """
    Performs convex optimization to find an optimal logit matrix Theta (Theta_opt)
    given an adjacency matrix A, a regularization parameter lambda (lam),
    and bounds for off-diagonal (c_off_diag) and diagonal (c_diag) entries.

    Args:
        A (np.ndarray): The input adjacency matrix (n x n).
        lam (float): The regularization parameter for the trace term.
        c_off_diag (float): Upper bound for the absolute values of off-diagonal entries of Theta.
        c_diag (float): Upper bound for the absolute values of diagonal entries of Theta.

    Returns:
        tuple: A tuple containing:
            - Theta_opt (np.ndarray): The optimized symmetric, PSD matrix (n x n).
            - prob.value (float): The optimal objective function value.
    """
    n = A.shape[0]              
    
    # define variables
    # variable (symmetric, PSD)
    Theta = cp.Variable((n, n), symmetric=True)

    # mask to exclude diagonal
    I = np.eye(n)
    mask = 1.0 - I

    # objective components
    # linear term: sum_{i!=j} -A_{ij}*Theta_{ij}
    linear_term = cp.sum(cp.multiply(mask * (-A), Theta))

    # convex logistic term: sum_{i!=j} log(1+exp(Theta_{ij}))
    # cvxpy has the `logistic` atom which equals log(1+exp(x))
    logistic_mat = cp.logistic(Theta)      # matrix of log(1+exp(Theta_ij))
    logistic_term = cp.sum(cp.multiply(mask, logistic_mat))

    # even though the theoretical analysis proceeds with lam = 44 * np.sqrt(n), 
    # in practice 2||A - P||_op is about 2 * np.sqrt(n), and this suffices in practice
    lam = max(2 * np.sqrt(n), 0.0001) if lam is None else lam 
    trace_term = lam * cp.trace(Theta)

    obj = cp.Minimize(linear_term + logistic_term + trace_term)

    # constraints: PSD, off-diagonal bounds, diagonal bounds
    # default off-diagonal bound
    c_off_diag = alpha * 3 * (np.log(n))**0.5 if c_off_diag is None else c_off_diag
    # default diagonal bound
    c_diag = 2 * alpha * d**0.5 if c_diag is None else c_diag
    
    constraints = []
    constraints.append(Theta >> 0)                       # PSD
    constraints.append(cp.abs(cp.multiply(mask, Theta)) <= c_off_diag)   # off-diagonal box
    constraints.append(cp.abs(cp.diag(Theta)) <= c_diag)    # diagonal box

    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.SCS, verbose=verbose, max_iters=max_iters, eps=eps)

    if Theta.value is None:
        raise RuntimeError(f"Solver failed: status={prob.status}")

    if verbose:
        print("status:", prob.status)
        print("optimal objective:", prob.value)
    Theta_opt = Theta.value
    return Theta_opt, prob.value


########### PROJECTED GRADIENT DESCENT PROXY OF CONVEX OPTIMIZATION ###########
# instead of solving the semidefinite program, we can approximate it by running projected gradient descent on Theta.
# This leads to faster computation, though still slower than gradient descent on the rescaled feature matrix Z.

# ---------- helpers ----------

def softplus(X):
    # stable: log(1+exp(x))
    return np.log1p(np.exp(-np.abs(X))) + np.maximum(X, 0.0)

def project_psd(Y):
    # project onto PSD cone by zeroing out negative eigenvalues
    Y = (Y + Y.T) / 2.0
    vals, vecs = np.linalg.eigh(Y) # the slow step when n is large
    vals_pos = np.clip(vals, 0.0, None)
    return (vecs * vals_pos) @ vecs.T

def clip_box(Y, c_off_diag, c_diag, verbose=False):
    # off-diagonals: |Theta_ij| <= c1
    # diagonals:     |Theta_ii| <= c2

    if verbose:
        # Check and print if any off-diagonal or diagonal entries reach the bounds
        off_diag_at_bound = np.any(np.abs(Y[np.triu_indices_from(Y, k=1)]) >= c_off_diag)
        diag_at_bound = np.any(np.abs(np.diag(Y)) >= c_diag)

        if off_diag_at_bound:
            print(f"Warning: Some off-diagonal entries reached the bound c1 = {c_off_diag}")
        if diag_at_bound:
            print(f"Warning: Some diagonal entries reached the bound c2 = {c_diag}")

    Y_clipped = np.copy(Y)
    n = Y.shape[0]
    I = np.eye(n, dtype=bool)
    off = ~I
    Y_clipped[off] = np.clip(Y_clipped[off], -c_off_diag, c_off_diag)
    diag = np.clip(np.diag(Y_clipped), -c_diag, c_diag)
    np.fill_diagonal(Y_clipped, diag)
    return Y_clipped

def pgd_random_initialization(n, d, alpha):
    X = get_latent_X(n, d)
    sigmoid_arg = alpha * (d**(-0.5)) * X @ X.T
    return sigmoid_arg
    

# ---------- objective and gradient ----------

def objective(Theta, A, lam, mask):
    # F(Theta) = sum_{i≠j} (-A_ij * Theta_ij + softplus(Theta_ij)) + lam * trace(Theta)
    off = mask.astype(bool)
    term_lin = -A[off] * Theta[off]
    term_sp = softplus(Theta[off])
    return float(np.sum(term_lin + term_sp) + lam * np.trace(Theta))

def grad_F(Theta, A, lam, mask):
    # off-diagonal: -A_ij + sigmoid(Theta_ij)
    # diagonal:     lambda
    S = sigmoid(Theta)
    G = np.zeros_like(Theta)

    off = mask.astype(bool)
    G[off] = -A[off] + S[off]

    diag_idx = np.arange(Theta.shape[0])
    G[diag_idx, diag_idx] = lam

    # enforce symmetry (for numerical safety)
    return (G + G.T) / 2.0


# ---------- projected gradient descent proxy of convex optimization ----------

def projected_gradient_descent(A, lam=None, c_off_diag=None, c_diag=None,
                               Theta0=None, max_iters=500, tol=1e-5,
                               eta0=0.25, backtracking=True, verbose=False):
    """
    Performs projected gradient descent to optimize the objective function related to the graph adjacency matrix A. 
    It finds a symmetric, PSD matrix Theta subject to box constraints on its entries.

    Args:
        A (np.ndarray): The input adjacency matrix (n x n).
        lam (float): The regularization parameter for the trace term. Default value calculated below.
        c_off_diag (float): Upper bound for the absolute values of off-diagonal entries of Theta. Default value calculated below.
        c_diag (float): Upper bound for the absolute values of diagonal entries of Theta. Default value calculated below.
        Theta0 (np.ndarray): Initial feasible point for Theta. If None, initializes with a random matrix.
        max_iters (int): Maximum number of gradient descent iterations. Defaults to 500.
        tol (float): Tolerance for stopping criteria (gradient norm or objective change). Defaults to 1e-5.
        eta0 (float): Initial step size. Defaults to 0.25.
        backtracking (bool): Whether to use backtracking line search. Defaults to True.
        verbose (bool): Whether to print iteration details. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - Theta (np.ndarray): The optimized symmetric, PSD matrix (n x n).
            - obj (float): The final objective function value.
    """
    n = A.shape[0]
    mask = np.ones((n, n)) - np.eye(n)

    # randomly initial Theta
    Theta0 = pgd_random_initialization(n, 10, alpha) if Theta0 is None else Theta 0
    
    Theta = (Theta0 + Theta0.T) / 2.0
    Theta = clip_box(Theta, c_off_diag, c_diag, verbose)
    Theta = project_psd(Theta)

    # default lambda                               
    lam = max(2 * np.sqrt(n), 0.0001) if lam is None else lam 
    # default off-diagonal bound
    c_off_diag = alpha * 3 * (np.log(n))**0.5 if c_off_diag is None else c_off_diag
    # default diagonal bound
    c_diag = 2 * alpha * d**0.5 if c_diag is None else c_diag

    obj = objective(Theta, A, lam, mask)
    if verbose:
        print(f"iter 0  obj {obj:.6e}")
    eta = eta0

    for k in range(1, max_iters+1):
        G = grad_F(Theta, A, lam, mask)
        grad_norm_sq = np.sum(G**2)
        last_obj = obj

        if backtracking:
            alpha = 1e-4
            beta = 0.5
            eta_local = eta
        
            accepted = False
            best_Y = None
            best_obj = np.inf
        
            for _ in range(20):
                Y = Theta - eta_local * G
                Y = clip_box(Y, c_off_diag, c_diag)
                Y = project_psd(Y)
                obj_new = objective(Y, A, lam, mask)
        
                # keep track of best attempt (useful if we fail Armijo)
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_Y = Y
        
                if obj_new <= obj - alpha * eta_local * grad_norm_sq:
                    accepted = True
                    break
        
                eta_local *= beta
        
            if not accepted:
                # accept best found if it improves, else stop.
                if best_obj < obj:
                    if verbose:
                        print(f"  [bt] Armijo failed; accepting best decrease (obj {obj:.6e} -> {best_obj:.6e}), eta {eta_local:.2e}")
                    Theta, obj, eta = best_Y, best_obj, eta_local
                else:
                    if verbose:
                        print(f"  [bt] Armijo failed and no decrease found; stopping (eta {eta_local:.2e})")
                    break
            else:
                Theta, obj, eta = Y, obj_new, eta_local
        else:
            Y = Theta - eta * G
            Y = clip_box(Y, c_off_diag, c_diag, verbose)
            Y = project_psd(Y)
            Theta, obj = Y, objective(Y, A, lam, mask)


        if verbose and (k % max(1, max_iters//20) == 0 or k <= 10):
            print(f"iter {k:3d}  obj {obj:.6e}  eta {eta:.2e}  ||grad||^2 {grad_norm_sq:.3e}")

        if grad_norm_sq <= tol**2:
            if verbose:
                print("stopping: gradient norm small")
            break

        if np.abs(obj - last_obj) <= tol * Theta.shape[0]**2: # the objective function of the pgd result roughly grows like n^2
            if verbose:
                print("stopping: objective change small")
            break

        if eta <= tol:
            if verbose:
                print("stopping: step size small")
            break


    return Theta, obj
