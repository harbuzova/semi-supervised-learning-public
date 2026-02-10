########### DATA GENERATION FUNCTIONS ###########

import pickle
import numpy as np
import random

def get_latent_X(n, d):
    """
    Given integers n and d, returns an n by d matrix where each entry is iid sampled from N(0,1).

    Args:
        n: The number of rows.
        d: The number of columns.

    Returns:
        A numpy array representing an n by d matrix with entries sampled from N(0,1).
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive integers")
    return np.random.randn(n, d)

def get_beta(d):
    """
    Given an integer d, returns a d-dimensional vector sampled from N(0, Id),
    and normalized to have norm 1.

    Args:
        d: The dimension of the vector.

    Returns:
        A numpy array representing the normalized d-dimensional vector.
    """
    if d <= 0:
        raise ValueError("d must be a positive integer")

    vector = np.random.randn(d)
    norm = np.linalg.norm(vector)

    # Resample if the norm is 0 to ensure the normalized vector has norm 1
    while norm <= 0.1:
        vector = np.random.randn(d)
        norm = np.linalg.norm(vector)

    return vector / norm

def generate_y(X, beta, gamma):
    """
    Given a n by d matrix X and d-dimensional vector beta, returns two n-dimensional
    vectors f and Y_noisy which is X * beta, and X * beta plus iid gaussian noise ~N(0,gamma^2) on each entry.

    Args:
        X: A numpy array representing the n by d matrix.
        beta: A numpy array representing the d-dimensional vector.
        gamma: A scalar representing the standard deviation of the Gaussian noise.

    Returns:
        A tuple of two numpy arrays representing the true n-dimensional vector f
        and the noisy observation Y_noisy.
    """
    if X.shape[1] != beta.shape[0]:
        raise ValueError("The number of columns in X must match the dimension of beta")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    n = X.shape[0]
    noise = np.random.normal(0, gamma, n)
    f = X @ beta # Matrix multiplication
    Y_noisy = f + noise  # Adding noise

    return f, Y_noisy

def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def generate_sigmoid_arg(X, alpha):
    n, d = X.shape
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive integers")

    # Calculate XX^T
    XXT = X @ X.T

    # Calculate the argument for the sigmoid function
    sigmoid_arg = alpha * (d**(-0.5)) * XXT
    return sigmoid_arg

def generate_probability_matrix(X, alpha):
    # Calculate the argument for the sigmoid function
    sigmoid_arg = generate_sigmoid_arg(X, alpha)

    # Apply the sigmoid function to get probabilities
    probabilities = sigmoid(sigmoid_arg)
    return probabilities


def generate_RGG(probabilities):
    n = probabilities.shape[0]
    # Sample from Bernoulli distribution for the upper triangle and diagonal
    upper_triangle_indices = np.triu_indices(n)
    G_upper = np.random.binomial(1, probabilities[upper_triangle_indices])

    # Create the symmetric matrix G
    G = np.zeros((n, n))
    G[upper_triangle_indices] = G_upper
    G = G + G.T - np.diag(np.diag(G)) # Add the lower triangle by transposing and subtract the diagonal to avoid doubling it

    return G

def random_subset(n, m):
    """
    Given integers n and m, returns a random subset of size m from the set {0, 1, ..., n-1}.

    Args:
        n: The size of the original set {0, 1, ..., n-1}.
        m: The size of the subset to be chosen.

    Returns:
        A list representing a random subset of size m from the set {0, 1, ..., n-1}.
    """
    if m < 0 or m >= n:
        raise ValueError("m must be between 0 and n-1 (inclusive)")
    return random.sample(range(0, n), m)


def generate_all_data(n, d, m, gamma, alpha, save_to_pkl=False, pkl_name="generated_data.pkl"):
    """
    Given integers n, d, m and scalars gamma, alpha as inputs.
    Generates a n by d matrix X, a d-dimensional vector beta,
    an n-dimensional vector Y which is X * beta
    and a n by n matrix G where each entry G_{ij} is independently sampled
    from Bernoulli(sigmoid(alpha * (d^{-1/2}) * (XX^T)_{ij})).
    Returns the first m entries of Y plus iid gaussian noise ~N(0,gamma^2) on each entry as the labeled data.

    Args:
      n: The number of rows.
      d: The number of columns.
      m: The number of labeled data points.
      gamma: A scalar representing the standard deviation of the Gaussian noise.
      alpha: A scalar constant.

    Returns:
      A dictionary containing:
        - "X": A numpy array representing the n by d matrix X.
        - "beta": A numpy array representing the d-dimensional vector beta.
        - "f": A numpy array representing the n-dimensional vector f.
        - "Theta": A numpy array representing the probabilities used to generate G.
        - "G": A numpy array representing the n by n matrix G.
        - "labeled_data": A numpy array representing the first m entries of f plus noise as the labeled data.
    """
    if save_to_pkl and not pkl_name.endswith('.pkl'):
        raise ValueError("pkl_name must end with .pkl if save_to_pkl is True")
    if m > n:
        raise ValueError("m must be less than or equal to n")
    

    X = get_latent_X(n, d)
    beta = get_beta(d)
    f, Y_noisy = generate_y(X, beta, gamma)
    sigmoid_arg = generate_sigmoid_arg(X, alpha)
    Theta = generate_probability_matrix(X, alpha)
    G = generate_RGG(Theta)
    S = random_subset(n, m)

    data_dict = {
        "X": X,
        "beta": beta,
        "f": f,
        "sigmoid_arg": sigmoid_arg,
        "Theta": Theta,
        "G": G,
        "S": S,
        "labeled_data": Y_noisy[S]
    }

    if save_to_pkl:
        with open(pkl_name, 'wb') as f:
            pickle.dump(data_dict, f)

    return data_dict

def print_S_submatrix_smallest_largest_singular_values(X, S):
    # Select the subset of rows from the matrix X based on the indices in S
    X_S = X[S, :]

    # Calculate the singular values of X_S
    singular_values = np.linalg.svd(X_S, compute_uv=False)

    # Normalize the singular values by dividing by sqrt(n/m)
    normalization_factor = np.sqrt(X.shape[0] / len(S))
    normalized_singular_values = singular_values * normalization_factor

    # Find the smallest and largest normalized singular values
    smallest_normalized_singular_value = np.min(normalized_singular_values)
    largest_normalized_singular_value = np.max(normalized_singular_values)

    print(f"Smallest normalized singular value of M_S: {smallest_normalized_singular_value}")
    print(f"Largest normalized singular value of M_S: {largest_normalized_singular_value}")




########### LINEAR REGRESSION ###########

def linear_regression(X, Y_S, S):
    """
    Performs linear regression using a subset of the data and predicts values for the entire dataset.

    Args:
        X: The full feature matrix (n x d).
        Y_S: The observed values for the subset of rows (m x 1).
        S: A list or array of indices indicating which rows of X are included in the subset.

    Returns:
        A numpy array representing the predicted values (f_hat) for the entire dataset (n x 1).
    """

    m = Y_S.shape[0]
    n, d = X.shape

    if m > n:
        raise ValueError("The number of observed values Y_S cannot exceed the number of rows in X")

    # Select the subset of rows from the matrix X based on row_indices
    X_S = X[S, :]

    beta_hat, *_ = np.linalg.lstsq(X_S, Y_S, rcond=rcond)  # solves min ||X_S beta - Y_S||
    f_hat = X @ beta_hat

    return f_hat


########### SPECTRAL DECOMPOSITION AND EIG FIX ###########


def adjust_recovered_matrix(M, d):
    # the step before eig_fix
    U, s, V = np.linalg.svd(M)
    return U[:, :d]


def eig_fix(B, S, c_lo = None, c_hi = None):

    # c_lo and c_hi are both Theta(1)
    # the normalization is taken care of in the algorithm

    n, d = B.shape
    m = S.shape[0]
    if m > n:
        raise ValueError("m must be less than or equal to n")
    B_S = B[S, :]
    U, s, V = np.linalg.svd(B_S)

    # Substitute the default lower bound c_lo if None
    c_lo = 0.5 * np.sqrt(m / n) if c_lo is None else c_lo

    # Substitute the default upper bound c_hi if None
    c_hi = 2 * np.sqrt(m / n) if c_hi is None else c_hi

    # Check if any sigular values of B_S lies outside [c_lo, c_hi]
    sigular_val_at_low = np.any(s < c_lo)
    sigular_val_at_up = np.any(s > c_hi)

    if sigular_val_at_low:
        print(f"Notice: eig_fix clips sigular values at lower bound = {c_lo}")
    if sigular_val_at_up:
        print(f"Notice: eig_fix clips sigular values at upper bound = {c_hi}")

    # Create a copy of the singular_values_A_S array
    modified_singular_values = np.copy(s)

    # Iterate through the modified_singular_values array and apply conditions
    modified_singular_values = np.clip(s, c_lo, c_hi)

    # Create D1 as a diagonal matrix with shape (m, d)
    # We need to pad the singular values with zeros if m != d
    D1 = np.zeros((m, d))
    min_dim = min(m, d)
    D1[:min_dim, :min_dim] = np.diag(modified_singular_values)

    B_S1 = U @ D1 @ V

    B1 = np.copy(B)

    # Use the list of indices S_indices to access and replace the corresponding rows in A1 with the rows from A_S1
    B1[S, :] = B_S1

    return B1


########### REGRESS (ALGORITHM 2) ###########


def regress(M, S, Y_S, c_lo = None, c_hi = None):
    """
    Algorithm 2 in our paper. 
    Estimates latent dimension d and labels f
    """
    n = M.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # estimate d by the number of eigenvalues that are larger than 0.5 * n
    d_hat = np.sum(eigenvalues > 0.5 * n)

    # assemble the d largest eigenvectors of M as a n by d matrix
    # np.linalg.eigh returns eigenvalues in ascending order, so we sort indices in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    B = eigenvectors[:, sorted_indices[:d_hat]]

    # run eig_fix on B
    B_fix = eig_fix(B, S, c_lo, c_hi)

    # regress on B_fix
    f_hat = linear_regression(B_fix, Y_S, S)

    return d_hat, f_hat
