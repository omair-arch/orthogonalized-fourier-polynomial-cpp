import numpy as np
import trimesh
import pyvista as pv
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from itertools import combinations_with_replacement
from os import path

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

def generate_eigenproducts_streaming(Phi, order=2):
    n_vertices, n_funcs = Phi.shape
    yield Phi[:, 0]
    Basis = Phi[:, 1:]
    for degree in range(1, order + 1):
        combos = combinations_with_replacement(range(Basis.shape[1]), degree)
        for combo in combos:
            prod = np.ones(n_vertices)
            for idx in combo:
                prod *= Basis[:, idx]
            yield prod

def orthogonalize_basis_streaming(A, Basis_generator, tol=1e-9):
    Ortho = []
    i = 0
    for vec in Basis_generator:
        print('i', i)
        i += 1
        q = vec.copy()
        for j in range(len(Ortho)):
            coeff = Ortho[j].T @ (A @ q)
            q -= coeff * Ortho[j]
        norm_q = np.sqrt(q.T @ (A @ q))
        if norm_q > tol:
            q /= norm_q
            Ortho.append(q)
    return np.column_stack(Ortho)

def create_pv_mesh(mesh, scalar_field):
    faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).astype(np.int32)
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    pv_mesh.point_data['Error'] = scalar_field
    return pv_mesh

# Load mesh
print("Loading mesh...")
mesh = trimesh.load_mesh(path.join(path.dirname(__file__), "../Meshes/205-Greek_Sculpture.off"), file_type='off')

# FEM & eigenfunctions
print("Computing FEM and eigendecomposition...")
S, A = compute_fem_matrices(mesh.vertices, mesh.faces)
K = 18
N = 3
eigvals, Phi = eigsh(S, k=N * K, M=A, sigma=-1e-5)

print("Computing NK eigenfunctions...")
Emb_NK = Phi[:, :N * K].T @ (A @ mesh.vertices)
FRec_NK = Phi[:, :N * K] @ Emb_NK
Diff_NK = np.linalg.norm(mesh.vertices - FRec_NK, axis=1)

print("Computing K eigenfunctions...")
Emb_K = Phi[:, :K].T @ (A @ mesh.vertices)
FRec_K = Phi[:, :K] @ Emb_K
Diff_K = np.linalg.norm(mesh.vertices - FRec_K, axis=1)

print("Computing Eigenproducts and Orthogonalization (Streaming-safe)...")
generator = generate_eigenproducts_streaming(Phi[:, :K], order=N)
Q = orthogonalize_basis_streaming(A, generator, tol=1e-9)
Emb_Q = Q.T @ (A @ mesh.vertices)
FRec_Q = Q @ Emb_Q
Diff_Q = np.linalg.norm(mesh.vertices - FRec_Q, axis=1)

# Create PyVista meshes
mesh_NK = create_pv_mesh(mesh, Diff_NK)
mesh_K = create_pv_mesh(mesh, Diff_K)
mesh_Q = create_pv_mesh(mesh, Diff_Q)

# Interactive multi-view window with PyVista
print("Plotting interactively in a single window...")

plotter = pv.Plotter(shape=(1, 3), border=True, window_size=(1800, 600))

plotter.subplot(0, 0)
plotter.add_mesh(mesh_NK, scalars='Error', cmap='hot', smooth_shading=True, scalar_bar_args={'title': 'NK Eigs'})
plotter.add_text(f"NK Eigs", font_size=12)

plotter.subplot(0, 1)
plotter.add_mesh(mesh_K, scalars='Error', cmap='hot', smooth_shading=True, scalar_bar_args={'title': 'K Eigs'})
plotter.add_text(f"K Eigs", font_size=12)

plotter.subplot(0, 2)
plotter.add_mesh(mesh_Q, scalars='Error', cmap='hot', smooth_shading=True, scalar_bar_args={'title': 'Ortho Basis'})
plotter.add_text(f"Ortho Basis", font_size=12)

plotter.show_axes()
plotter.show()