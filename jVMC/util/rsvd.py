import jax
import jax.numpy as jnp
from jax import random, lax
from jax.scipy.linalg import svd
from jax import random

def randomized_eigh(A, k, n_oversamples=10, n_iter=2, key=None, which='LM'):
    """
    Randomized eigendecomposition for symmetric/Hermitian matrices.

    Args:
        A: Input symmetric matrix (n x n)
        k: Number of eigenvalues/eigenvectors to compute
        n_oversamples: Additional samples for improved accuracy (default: 10)
        n_iter: Number of power iterations (default: 2)
        key: JAX PRNG key (if None, creates a new key)
        which: Selection criterion ('LM' for largest magnitude, 'LA' for largest algebraic)

    Returns:
        w: Array of top-k eigenvalues
        V: Matrix of corresponding eigenvectors (each column is an eigenvector)
    """
    if key is None:
        key = random.PRNGKey(0)

    n = A.shape[0]
    total_rank = min(k + n_oversamples, n)

    # Generate random test matrix
    key, subkey = random.split(key)
    dtype = A.dtype
    Omega = random.normal(subkey, (n, total_rank), dtype=dtype)

    # Power iterations
    Y = A @ Omega
    for _ in range(n_iter):
        # Orthonormalize to maintain stability
        Q = jnp.linalg.qr(Y, mode='reduced')[0]
        Y = A @ Q

    # Final orthonormal basis
    Q = jnp.linalg.qr(Y, mode='reduced')[0]

    # Form the small projected matrix
    B = Q.T @ A @ Q

    # Solve the small eigenvalue problem
    w_small, V_small = jnp.linalg.eigh(B)

    # Select eigenvalues/eigenvectors based on criterion
    if which == 'LM':  # Largest magnitude
        idx = jnp.argsort(jnp.abs(w_small))[::-1][:k]
    elif which == 'LA':  # Largest algebraic
        idx = jnp.argsort(w_small)[::-1][:k]
    else:
        raise ValueError("which must be 'LM' or 'LA'")

    w = w_small[idx]
    V_small = V_small[:, idx]

    # Project eigenvectors back to original space
    V = Q @ V_small

    return w, V


def adaptive_randomized_eigh(A, tol=1e-6, max_rank=200, block_size=10, 
                            n_iter=2, key=None, verbose=False):
    """
    Adaptive randomized eigendecomposition for symmetric matrices.
    
    Iteratively increases the rank until the residual norm is below tolerance.
    
    Args:
        A: Symmetric input matrix (n x n)
        tol: Tolerance for residual norm (default: 1e-6)
        max_rank: Maximum rank to compute (default: 100)
        block_size: Number of vectors to add per iteration (default: 10)
        n_iter: Number of power iterations per step (default: 2)
        key: JAX PRNG key (if None, creates new key)
        verbose: Print progress information (default: False)
        
    Returns:
        w: Array of eigenvalues (in descending order)
        V: Matrix of corresponding eigenvectors
        residual_norm: Final residual norm
        total_rank: Final rank used
    """
    if key is None:
        key = random.PRNGKey(0)
    
    n = A.shape[0]
    dtype = A.dtype
    min_rank = min(block_size, max_rank)
    
    # Initialize basis and projection matrix
    key, subkey = random.split(key)
    Q = random.normal(subkey, (n, min_rank), dtype=dtype)
    total_rank = min_rank
    
    # Define helper functions
    def qr_safe(X):
        """Safe QR decomposition with reduced mode."""
        Q, R = jnp.linalg.qr(X, mode='reduced')
        return Q, R
    
    def power_iteration(Q, A, n_iter):
        """Perform power iterations to improve basis quality."""
        def body_fn(_, Qi):
            Qi = A @ Qi
            Qi, _ = qr_safe(Qi)
            return Qi
        return lax.fori_loop(0, n_iter, body_fn, Q)
    
    def update_basis(Q, A, total_rank, block_size, key, n_iter):
        """Extend the basis with new vectors and apply power iterations."""
        # Generate new random vectors
        key, subkey = random.split(key)
        Omega = random.normal(subkey, (n, block_size), dtype=dtype)
        
        # Orthogonalize against current basis
        if total_rank > 0:
            Omega = Omega - Q @ (Q.T @ Omega)
        
        # Orthonormalize new vectors
        Q_new, _ = qr_safe(Omega)
        actual_block_size = Q_new.shape[1]
        
        # Combine bases
        Q_ext = jnp.hstack([Q, Q_new]) if total_rank > 0 else Q_new
        total_rank_ext = total_rank + actual_block_size
        
        # Apply power iterations to the extended basis
        Q_ext = power_iteration(Q_ext, A, n_iter)
        return Q_ext, total_rank_ext, key, actual_block_size
    
    def compute_eigenproblem(Q, A):
        """Compute the projected eigenproblem."""
        B = Q.T @ (A @ Q)
        w, V_small = jnp.linalg.eigh(B)
        # Sort in descending order
        idx = jnp.argsort(w)[::-1]
        w = w[idx]
        V_small = V_small[:, idx]
        V_full = Q @ V_small
        return w, V_full
    
    def compute_residual(A, V, w):
        """Compute residual norm."""
        R = A @ V - V * w
        return jnp.linalg.norm(R, 'fro')
    
    # Main iteration loop
    residual_norm = jnp.inf
    iteration = 0
    converged = False
    
    while not converged and total_rank < max_rank:
        iteration += 1
        
        # Extend basis and apply power iterations
        Q, total_rank, key, actual_block_size = update_basis(
            Q, A, total_rank, block_size, key, n_iter
        )
        
        # Compute current eigenproblem
        w, V = compute_eigenproblem(Q, A)
        
        # Compute residual norm
        residual_norm = compute_residual(A, V, w)
        
        # Check convergence
        converged = residual_norm < tol
        
        if verbose:
            print(f"Iter {iteration}: Rank {total_rank}, Residual {residual_norm:.4e}")
        
        # Break if we've reached max rank
        if total_rank >= max_rank:
            if verbose:
                print(f"Reached max rank {max_rank}")
            break
    
    # Final computation with converged basis
    w, V = compute_eigenproblem(Q, A)
    residual_norm = compute_residual(A, V, w)
    
    if verbose:
        print(f"Final rank: {total_rank}, Final residual: {residual_norm:.4e}")
    
    return w, V, residual_norm, total_rank
