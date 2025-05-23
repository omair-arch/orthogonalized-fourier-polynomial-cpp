import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from common_functions import FEM_higher, eigprods, orthogonalize_basis, compute_dirichlet
from os import path

# -------------------------
# Parameters
# -------------------------
SrcFile = "../Meshes/sheep.off"
KSrc = 15
N = 3
OrthoThreshReiter = 1e-9
tau1 = 1e-2
tau2 = 0.0025
# def compute_fem_matrices(vertices, faces):
#     n_vertices = vertices.shape[0]
#     L = csr_matrix((n_vertices, n_vertices)).tolil()
#     M = np.zeros(n_vertices)
#     for tri in faces:
#         i, j, k = tri
#         vi, vj, vk = vertices[i], vertices[j], vertices[k]
#         e0, e1, e2 = vj - vk, vk - vi, vi - vj
#         cot0 = np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))
#         cot1 = np.dot(e2, e0) / np.linalg.norm(np.cross(e2, e0))
#         cot2 = np.dot(e0, e1) / np.linalg.norm(np.cross(e0, e1))
#         L[i, j] += cot2
#         L[j, i] += cot2
#         L[j, k] += cot0
#         L[k, j] += cot0
#         L[k, i] += cot1
#         L[i, k] += cot1
#         area = np.linalg.norm(np.cross(vj - vi, vk - vi)) / 6.0
#         M[i] += area
#         M[j] += area
#         M[k] += area
#     L = -0.5 * (L + L.T)
#     M = diags(M)
#     return L.tocsr(), M.tocsr()
# -------------------------
# Load and prepare the mesh
# -------------------------
print("Loading the mesh...")
Src = trimesh.load_mesh(path.join(path.dirname(__file__), SrcFile), file_type='off')
Src_points = Src.vertices
Src_faces = Src.faces

# Normalize the mesh vertices by surface area
v1 = Src_points[Src_faces[:, 0]]
v2 = Src_points[Src_faces[:, 1]]
v3 = Src_points[Src_faces[:, 2]]
areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
Src_points /= np.sqrt(np.sum(areas))

# -------------------------
# FEM & eigen decomposition
# -------------------------
print("Solving FEM and eigenproblem...")
S, A, _ = FEM_higher({'VERT': Src_points, 'TRIV': Src_faces, 'n': Src_points.shape[0], 'm': Src_faces.shape[0]}, 1, 'Dirichlet')
eigvals, Phi = eigsh(S, k=N * KSrc, M=A, sigma=-1e-5)
Phi = np.real(Phi)

# -------------------------
# Function to filter
# -------------------------
f = Src_points.copy()  # XYZ coordinates

# -------------------------
# Transfer with eigenproducts (first filter)
# -------------------------
print("Computing eigenproducts...")
PolyPhi = eigprods({'A': A, 'Phi': Phi}, KSrc - 1, N, normalized=True)
freqs = compute_dirichlet(S, PolyPhi)
order = np.argsort(freqs)
basis_poly = PolyPhi[:, order]

filter1 = np.exp(-tau1 * np.abs(np.arange(basis_poly.shape[1])))
filter1 /= np.max(filter1)
coeff1 = np.linalg.pinv(np.sqrt(A.toarray()) @ basis_poly) @ (np.sqrt(A.toarray()) @ f)
prod1 = basis_poly @ (filter1[:, None] * coeff1)

# -------------------------
# Transfer with orthogonalized eigenproducts (first filter)
# -------------------------
print("Computing orthogonal basis...")
QS = orthogonalize_basis(A, PolyPhi, tol=OrthoThreshReiter, adjust=True)
freqs = compute_dirichlet(S, QS)
order = np.argsort(freqs)
basis_our = QS[:, order]
coeff1_ortho = basis_our.T @ (A @ f)
ortho1 = basis_our @ (filter1[:, None] * coeff1_ortho)

# -------------------------
# Transfer with eigenproducts (second filter)
# -------------------------
filter2 = np.exp(-tau2 * (np.abs(basis_poly.shape[1] - 1 - np.arange(basis_poly.shape[1]))))
filter2 /= np.max(filter2)
coeff2 = np.linalg.pinv(np.sqrt(A.toarray()) @ basis_poly) @ (np.sqrt(A.toarray()) @ f)
prod2 = basis_poly @ (filter2[:, None] * coeff2)

# -------------------------
# Transfer with orthogonalized eigenproducts (second filter)
# -------------------------
coeff2_ortho = basis_our.T @ (A @ f)
ortho2 = basis_our @ (filter2[:, None] * coeff2_ortho)

# -------------------------
# PyVista Visualization
# -------------------------
def show_mesh(vertices, faces, title):
    mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
    mesh["colors"] = np.ones((vertices.shape[0],)) * 1.0  # White base
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', show_edges=False)
    plotter.add_text(title, font_size=14)
    plotter.view_vector([0, 1, 0])
    plotter.show()

# First row
show_mesh(Src_points, Src_faces, "Source (Original)")
show_mesh(prod1, Src_faces, "Poly (Filter 1)")
show_mesh(ortho1, Src_faces, "Ortho (Filter 1)")

# Filter plot 1
plt.figure(figsize=(8, 4))
plt.plot(filter1, linewidth=2)
plt.title("Filter 1")
plt.show()

# Second row
show_mesh(Src_points, Src_faces, "Source (Original)")
show_mesh(prod2, Src_faces, "Poly (Filter 2)")
show_mesh(ortho2, Src_faces, "Ortho (Filter 2)")

# Filter plot 2
plt.figure(figsize=(8, 4))
plt.plot(filter2, linewidth=2)
plt.title("Filter 2")
plt.show()
