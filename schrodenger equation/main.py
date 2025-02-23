import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

"""
 NAME: Baaman Cletus
 INDEX NO.: 8660621"""
# Define parameters
L = 1.0  # Well width
N = 100  # Number of grid points
dx = L / (N + 1)  # Grid spacing
hbar = 1.0  # Reduced Planck's constant (in atomic units)
m = 1.0  # Particle mass (in atomic units)

# Construct the finite difference Hamiltonian matrix
diagonal = np.full(N, -2.0)
off_diagonal = np.full(N - 1, 1.0)
H = (-hbar ** 2 / (2 * m * dx ** 2)) * (np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1))

# Solve for eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)

# Normalize wave functions
wavefunctions = eigenvectors / np.sqrt(dx)

# Convert eigenvalues to energy levels
energy_levels = eigenvalues

# Plot the first few wavefunctions
x = np.linspace(0, L, N + 2)  # Include boundary points

plt.figure(figsize=(8, 6))
for i in range(3):  # Plot first three wavefunctions
    psi = np.zeros(N + 2)  # Include boundary conditions
    psi[1:N + 1] = wavefunctions[:, i]
    plt.plot(x, psi, label=f'ψ_{i + 1} (E={energy_levels[i]:.3f})')

plt.xlabel("x")
plt.ylabel("Wavefunction ψ(x)")
plt.title("Wavefunctions of the Infinite Potential Well")
plt.legend()
plt.grid()
plt.show()

# Print first few energy levels
print("First few energy levels (in arbitrary units):")
for i in range(3):
    print(f"E_{i + 1} = {energy_levels[i]:.3f}")
