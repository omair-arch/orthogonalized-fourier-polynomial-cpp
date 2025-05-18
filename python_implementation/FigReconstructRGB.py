import numpy as np
import pyvista as pv
import trimesh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from itertools import combinations_with_replacement

def compute_fem_matrices(vertices, faces):
    n_vertices = vertices.shape[0]
    L = csr_matrix((n_vertices, n_vertices)).tolil()
    M = np.zeros(n_vertices)
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        e0, e1, e2 = vj - vk, vk - vi, vi - vj
        cot0 = np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))
        cot1 = np.dot(e2, e0) / np.linalg.norm(np.cross(e2, e0))
        cot2 = np.dot(e0, e1) / np.linalg.norm(np.cross(e0, e1))
        L[i, j] += cot2
        L[j, i] += cot2
        L[j, k] += cot0
        L[k, j] += cot0
        L[k, i] += cot1
        L[i, k] += cot1
        area = np.linalg.norm(np.cross(vj - vi, vk - vi)) / 6.0
        M[i] += area
        M[j] += area
        M[k] += area
    L = -0.5 * (L + L.T)
    M = diags(M)
    return L.tocsr(), M.tocsr()

def compute_eigenproducts(Phi, order=2):
    n_vertices, n_funcs = Phi.shape
    PolyPhi = [Phi[:, 0]]
    Basis = Phi[:, 1:]
    for degree in range(1, order + 1):
        combos = combinations_with_replacement(range(Basis.shape[1]), degree)
        for combo in combos:
            prod = np.ones(n_vertices)
            for idx in combo:
                prod *= Basis[:, idx]
            PolyPhi.append(prod)
    return np.column_stack(PolyPhi)


# def orthogonalize_basis(A, Basis, tol=1e-9, reiterate=False):
#     Ortho = []
#     AQ_cache = []  # Cache A @ q_i for later reuse
#
#     A = A.tocsr()  # Ensure efficient row access
#     for i in range(Basis.shape[1]):
#         q = Basis[:, i].copy()
#         Aq = A @ q
#
#         # Project onto all previous orthogonal vectors
#         for j in range(len(Ortho)):
#             coeff = Ortho[j].T @ Aq
#             q -= coeff * Ortho[j]
#             Aq -= coeff * AQ_cache[j]  # Update Aq alongside q
#
#         norm_q = np.sqrt(q.T @ Aq)
#         if norm_q < tol:
#             continue
#         q /= norm_q
#         Aq /= norm_q
#
#         if reiterate:
#             proj = np.array([vec.T @ (A @ q) for vec in Ortho])
#             while np.any(np.abs(proj) > tol):
#                 for j in range(len(Ortho)):
#                     coeff = Ortho[j].T @ (A @ q)
#                     q -= coeff * Ortho[j]
#                 Aq = A @ q
#                 norm_q = np.sqrt(q.T @ Aq)
#                 if norm_q < tol:
#                     break
#                 q /= norm_q
#                 Aq /= norm_q
#                 proj = np.array([vec.T @ (A @ q) for vec in Ortho])
#             else:
#                 Ortho.append(q)
#                 AQ_cache.append(Aq)
#                 continue
#
#         Ortho.append(q)
#         AQ_cache.append(Aq)
#
#     return np.column_stack(Ortho) if Ortho else np.empty((Basis.shape[0], 0))

def orthogonalize_basis(A, Basis, tol=1e-9, reiterate=False):
    Ortho = []
    AQ_cache = []

    A = A.tocsr()
    n_vectors = Basis.shape[1]

    for i in range(n_vectors):
        q = Basis[:, i].copy()
        Aq = A @ q

        for j in range(len(Ortho)):
            coeff = np.dot(Ortho[j], Aq)
            q -= coeff * Ortho[j]
            Aq -= coeff * AQ_cache[j]

        norm_q = np.linalg.norm(Aq if not reiterate else A @ q)
        if norm_q < tol:
            continue

        q /= norm_q
        Aq /= norm_q

        if reiterate:
            while True:
                proj_coeffs = np.array([np.dot(vec, A @ q) for vec in Ortho])
                if np.all(np.abs(proj_coeffs) <= tol):
                    break
                for j, coeff in enumerate(proj_coeffs):
                    q -= coeff * Ortho[j]
                Aq = A @ q
                norm_q = np.linalg.norm(Aq)
                if norm_q < tol:
                    break
                q /= norm_q
                Aq /= norm_q

        Ortho.append(q)
        AQ_cache.append(Aq)

    return np.column_stack(Ortho) if Ortho else np.empty((Basis.shape[0], 0))


plotter = pv.Plotter(shape=(1, 4), border=True, window_size=(1800, 600))
def plot_reconstruction(row,col,mesh, vertices_colors, title):
    # pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]))
    # pv_mesh.point_data['RGB'] = np.clip(vertices_colors, 0, 1)
    # plotter = pv.Plotter()
    plotter.subplot(row,col)
    # Handle 1D or low dynamic range by forcing white
    if np.allclose(vertices_colors, 0) or np.abs(vertices_colors.max() - vertices_colors.min()) < 1e-8:
        FRec_norm = np.ones((mesh.vertices.shape[0], 3))  # pure white RGB
    else:
        # Normalize safely between 0-1
        FRec_norm = vertices_colors.copy()
        if FRec_norm.ndim == 1:
            FRec_norm = FRec_norm[:, np.newaxis]
        min_val = FRec_norm.min()
        max_val = FRec_norm.max()
        FRec_norm = (FRec_norm - min_val) / (max_val - min_val + 1e-8)
        if FRec_norm.shape[1] == 1:
            FRec_norm = np.repeat(FRec_norm, 3, axis=1)

    faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).astype(np.int32)
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    pv_mesh.point_data['RGB'] = FRec_norm
    plotter.add_mesh(pv_mesh, scalars='RGB', rgb=True, show_edges=False, smooth_shading=True)
    plotter.add_text(title, font_size=14)
    # plotter.show()

# ------------------- Main -------------------
print("Loading mesh and data...")
mesh = trimesh.load_mesh('../Meshes/bunny.off', file_type='off')
# mesh = trimesh.load_mesh('Meshes/bunny.off', file_type='off')
n_vertices = mesh.vertices.shape[0]

# Generate random RGB color for each vertex (values between 0 and 1)
F = np.random.rand(n_vertices, 3)

# Save it to .npy file
np.save('texture_bunny.npy', F)

print(f"Saved dummy texture_bunny.npy with shape {F.shape}")
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(90), [1, 0, 0]))
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(-10), [0, 0, 1]))

S, A = compute_fem_matrices(mesh.vertices, mesh.faces)
NumEigs = 200
K = 100
Order = 2

eigvals, Phi = eigsh(S, k=NumEigs, M=A, sigma=-1e-5)
F = np.load('texture_bunny.npy')  # assuming the RGB texture is loaded as Nx3 numpy array


print("Reconstruction using Eigenfunctions...")
FK = Phi[:, :K] @ (Phi[:, :K].T @ (A @ F))
plot_reconstruction(0,0,mesh, FK, f"{K} Eigs Reconstruction")

print("Reconstruction using Eigenproducts...")
PolyPhi = compute_eigenproducts(Phi[:, :K], order=Order)
EmbP = np.linalg.pinv(np.sqrt(A.toarray()) @ PolyPhi) @ (np.sqrt(A.toarray()) @ F)
FP = PolyPhi @ EmbP
plot_reconstruction(0,1,mesh, FP, "Eigenproducts Reconstruction")

print("Reconstruction using Ortho Small Threshold...")
Q_small = orthogonalize_basis(A, PolyPhi, tol=1e-9, reiterate=True)
FS = Q_small @ (Q_small.T @ (A @ F))
plot_reconstruction(0,2,mesh, FS, "Ortho Small Threshold")

print("Reconstruction using Ortho Large Threshold...")
Q_large = orthogonalize_basis(A, PolyPhi, tol=1e-4, reiterate=False)
FL = Q_large @ (Q_large.T @ (A @ F))
plot_reconstruction(0,3,mesh, FL, "Ortho Large Threshold")
plotter.show()