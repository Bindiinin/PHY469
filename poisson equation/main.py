import numpy as np
import matplotlib.pyplot as plt
import time

"""
NAME: BAAMAN CLETUS
INDEX NO.: 8660621
"""

def solve_poisson(N, max_iter=5000, tolerance=1e-5, plot=True):
    """Solves Poisson's equation and analyzes convergence behavior."""
    L = 1.0  # Domain size
    h = L / (N - 1)  # Grid spacing
    phi = np.zeros((N, N))  # Potential function
    rho = np.zeros((N, N))  # Source term

    # Adding charge distribution at the center
    rho[N // 2, N // 2] = 100

    # Dirichlet boundary conditions (fixed at zero)
    phi[:, 0] = phi[:, -1] = phi[0, :] = phi[-1, :] = 0

    start_time = time.time()
    error_list = []  # To track convergence

    for it in range(max_iter):
        phi_old = phi.copy()

        # Finite difference update
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                phi[i, j] = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1] - h ** 2 * rho[i, j])

        # Compute max error
        error = np.max(np.abs(phi - phi_old))
        error_list.append(error)

        if error < tolerance:
            break  # Stop if solution has converged

    execution_time = time.time() - start_time
    print(f'N = {N}, Max Iterations = {max_iter}, Converged in {it} iterations, Time taken: {execution_time:.4f} sec')

    if plot:
        # Plot solution
        plt.figure(figsize=(6, 5))
        plt.imshow(phi, extent=[0, L, 0, L], origin='lower', cmap='inferno')
        plt.colorbar(label='Potential φ(x,y)')
        plt.title(f'Poisson Equation Solution (N={N})')
        plt.show()

        # Plot error reduction
        plt.figure(figsize=(6, 4))
        plt.semilogy(error_list, label=f'N={N}')
        plt.xlabel('Iteration')
        plt.ylabel('Max Error')
        plt.title('Convergence Behavior')
        plt.legend()
        plt.grid()
        plt.show()

    return it, execution_time, error_list[-1]


# Compare convergence for different grid sizes and max iterations
grid_sizes = [20, 50, 100]
max_iters = [5000, 10000]  # Standard and extended iteration limits

results = []
for N in grid_sizes:
    for max_iter in max_iters:
        results.append((N, max_iter, solve_poisson(N, max_iter=max_iter, plot=True)))

# Print summary of results
for (N, max_iter, (iterations, time_taken, final_error)) in results:
    print(
        f'Grid Size: {N}, Max Iterations: {max_iter}, Converged in: {iterations}, Execution Time: {time_taken:.4f} sec, Final Error: {final_error:.6f}')