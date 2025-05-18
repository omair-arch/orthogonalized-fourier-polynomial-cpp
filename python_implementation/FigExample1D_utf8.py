import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# -------------------------
# Parameters
# -------------------------
MaxK = 4
PlotAnalytic = False
NumPlotPoints = 500  # Can be increased since it's now fast

# -------------------------
# Create the functions numerically using lambdas
# -------------------------
K = np.arange(0, MaxK + 1)
Phi_numeric = [lambda x, k=k: np.sin(k * x) for k in K]  # lambdas for Phi

# -------------------------
# Eigenproducts
# -------------------------
PolyPhi_numeric = []
for i in range(1, len(K)):
    for j in range(i, len(K)):
        PolyPhi_numeric.append(lambda x, i=i, j=j: Phi_numeric[i](x) * Phi_numeric[j](x))

# -------------------------
# Orthogonalization (Gram-Schmidt)
# -------------------------
Ortho_numeric = PolyPhi_numeric.copy()

def inner_product(f, g):
    return quad(lambda x: f(x) * g(x), 0, 2 * np.pi)[0]

# Gram-Schmidt process
for i in range(1, len(Ortho_numeric)):
    f = Ortho_numeric[i]
    combined_basis = Phi_numeric + Ortho_numeric[:i]
    Coeff = [inner_product(f, basis) for basis in combined_basis]
    def projection(x, Coeff=Coeff, basis=combined_basis):
        return sum(c * b(x) for c, b in zip(Coeff, basis))
    Ortho_numeric[i] = lambda x, f=f, projection=projection: f(x) - projection(x)

# Normalize
for i in range(len(Ortho_numeric)):
    norm = np.sqrt(inner_product(Ortho_numeric[i], Ortho_numeric[i]))
    Ortho_numeric[i] = lambda x, f=Ortho_numeric[i], norm=norm: f(x) / norm

# -------------------------
# Plotting
# -------------------------
X_vals = np.linspace(0, 2 * np.pi, NumPlotPoints)

def plot_function(f, X, color, pos, total, ylim_range=(-1, 1)):
    Y = f(X)
    plt.subplot(5, total, pos)
    plt.plot(X, Y, color, linewidth=2)
    plt.ylim(ylim_range)
    plt.xlim([0, 2 * np.pi])

plt.figure(figsize=(15, 10))

if PlotAnalytic:
    # Plot eigenfunctions
    for i, phi in enumerate(Phi_numeric):
        plot_function(phi, X_vals, 'b', i + 1, len(K))

    # Plot eigenproducts
    for i, poly in enumerate(PolyPhi_numeric):
        plot_function(poly, X_vals, 'r', i + 1 + len(K), len(K))

    # Plot orthogonalized eigenproducts
    for i, ortho in enumerate(Ortho_numeric):
        plot_function(ortho, X_vals, 'g', i + 1 + 3 * len(K), len(K))
else:
    # Same plotting for numerical case
    for i, phi in enumerate(Phi_numeric):
        plot_function(phi, X_vals, 'b', i + 1, len(K))

    for i, poly in enumerate(PolyPhi_numeric):
        plot_function(poly, X_vals, 'r', i + 1 + len(K), len(K))

    for i, ortho in enumerate(Ortho_numeric):
        plot_function(ortho, X_vals, 'g', i + 1 + 3 * len(K), len(K))

plt.tight_layout()
plt.show()
