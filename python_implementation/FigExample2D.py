
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import pyvista as pv
from itertools import combinations_with_replacement

# -------------------------
# Parameters
# -------------------------
MeshFile = '../Meshes/bunny.off'
NumRows = 1
NumCols = 5
OrthoThresh = 1e-9
FontSize = 24

# def compute_eigenproducts(Phi, order=2, include_linear=False):
#     n_vertices, n_funcs = Phi.shape
#     PolyPhi = []
#
#     if include_linear:
#         for i in range(n_funcs):
#             PolyPhi.append(Phi[:, i])
#
#     # Quadratic products
#     for i in range(n_funcs):
#         for j in range(i, n_funcs):
#             PolyPhi.append(Phi[:, i] * Phi[:, j])
#
#     return np.array(PolyPhi).T  # Columns are new functions

def compute_eigenproducts(Phi, A, order=2, normalized=False):
    n_vertices, n_funcs = Phi.shape

    # Always keep the constant function explicitly
    PolyPhi = [Phi[:, 0]]
    Basis = Phi[:, 1:]

    # Generate all multi-indices for the given order using combinations with replacement
    all_indices = []
    for degree in range(1, order + 1):
        all_indices.extend(combinations_with_replacement(range(Basis.shape[1]), degree))

    # Compute the products for all these indices
    for idx_combo in all_indices:
        product = np.ones(n_vertices)
        for idx in idx_combo:
            product *= Basis[:, idx]
        PolyPhi.append(product)

    # Stack all as columns
    PolyPhi = np.column_stack(PolyPhi)

    # Normalize if requested (using mass matrix A)
    if normalized:
        norms = np.sqrt(np.einsum('ij,ij->j', PolyPhi, A @ PolyPhi))
        PolyPhi = PolyPhi / norms

    return PolyPhi
# def compute_eigenproducts(Phi, A, order=2, normalized=False):
#     """
#     Compute n-th order polynomial eigenproducts from eigenfunctions Phi.
#     Optional normalization w.r.t mass matrix A.
#     """
#
#
#     n_vertices, n_funcs = Phi.shape
#     PolyPhi = [Phi[:, 0]]  # Keep the constant eigenfunction
#     Basis = Phi[:, 1:]     # Non-constant
#
#     Prods = Basis.copy()
#     PolyPhi.extend([Prods[:, i] for i in range(Prods.shape[1])])
#
#     # for i in range(2, order + 1):
#     #     tmp1 = np.tile(Basis, (1, Prods.shape[1]))
#     #     tmp2 = np.reshape(np.tile(Prods, (n_funcs, 1)), (n_vertices, n_funcs * Prods.shape[1]))
#     #
#     #     Prods = tmp1 * tmp2
#     #     PolyPhi.extend([Prods[:, j] for j in range(Prods.shape[1])])
#     for i in range(2, order + 1):
#         # Get current product count
#         current_prods = Prods.shape[1]
#
#         # Expand Basis to match Prods
#         tmp1 = np.tile(Basis, (1, current_prods))  # (n_vertices, n_funcs * current_prods)
#         tmp2 = np.tile(Prods, (1, n_funcs))
#
#         assert tmp1.shape == tmp2.shape, f"Mismatch: {tmp1.shape} vs {tmp2.shape}"
#
#         Prods = tmp1 * tmp2
#         PolyPhi.extend([Prods[:, j] for j in range(Prods.shape[1])])
#
#     # Select unique combinations like MATLAB's sub_index
#     PolyPhi = np.stack(PolyPhi, axis=1)
#     PolyPhi_sorted = np.sort(PolyPhi, axis=0)
#     _, unique_indices = np.unique(PolyPhi_sorted.T, axis=0, return_index=True)
#     PolyPhi = PolyPhi[:, np.sort(unique_indices)]
#
#     # Normalize if requested
#     if normalized:
#         norms = np.sqrt(np.einsum('ij,ij->j', PolyPhi, A @ PolyPhi))
#         PolyPhi = PolyPhi / norms
#
#     return PolyPhi

def compute_fem_matrices(vertices, faces, order=1, bc='Dirichlet'):
    """
    Simple cotangent Laplacian and lumped mass matrix approximation.
    Pure numpy + scipy implementation.
    """
    from scipy.sparse import lil_matrix, diags

    n_vertices = vertices.shape[0]
    L = lil_matrix((n_vertices, n_vertices))
    M = np.zeros(n_vertices)

    # Loop over faces to compute cotangent weights and area
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # Edge vectors
        e0 = vj - vk
        e1 = vk - vi
        e2 = vi - vj

        # Compute cotangents
        cot0 = np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))
        cot1 = np.dot(e2, e0) / np.linalg.norm(np.cross(e2, e0))
        cot2 = np.dot(e0, e1) / np.linalg.norm(np.cross(e0, e1))

        # Update Laplacian (symmetric)
        L[i, j] += cot2
        L[j, i] += cot2
        L[j, k] += cot0
        L[k, j] += cot0
        L[k, i] += cot1
        L[i, k] += cot1

        # Accumulate area (lumped mass approximation)
        area = np.linalg.norm(np.cross(vj - vi, vk - vi)) / 6.0
        M[i] += area
        M[j] += area
        M[k] += area

    # Negative Laplacian (cotangent Laplacian is usually -L)
    L = -0.5 * (L + L.T)
    # Lumped mass matrix as diagonal
    M = diags(M)

    return L.tocsr(), M.tocsr()


# def orthogonalize_basis(A, PolyPhi, thresh=1e-9, verbose=True):
#     Q = []
#     R = []
#     for i in range(PolyPhi.shape[1]):
#         q = PolyPhi[:, i].copy()
#         r = []
#         for j in range(len(Q)):
#             coeff = Q[j].T @ (A @ q)
#             q -= coeff * Q[j]
#             r.append(coeff)
#         norm_q = np.sqrt(q.T @ (A @ q))
#         # if norm_q > thresh:
#         q /= norm_q
#         Q.append(q)
#         r.append(norm_q)
#         # elif verbose:
#         #     print(f"Vector {i} discarded (norm below threshold)")
#     return np.array(Q).T, np.array(R)
def orthogonalize_basis(A, Basis, Tol=1e-9, Adjust=True, verbose=True):
    """
    Orthogonalize a set of basis functions using Gram-Schmidt with mass matrix A.
    Fully MATLAB-like.
    Returns Ortho, Coeff, SubIdx.
    """
    import numpy as np

    n_vectors = Basis.shape[1]
    Ortho = []
    Coeff = []
    SubIdx = []

    # First vector initialization
    norm0 = np.sqrt(Basis[:, 0].T @ (A @ Basis[:, 0]))
    q0 = Basis[:, 0] / norm0
    Ortho.append(q0)
    Coeff.append(np.array([norm0]))
    SubIdx.append(0)

    for i in range(1, n_vectors):
        new_proj = np.array([q.T @ (A @ Basis[:, i]) for q in Ortho])
        new_vec = Basis[:, i] - sum(Ortho[j] * new_proj[j] for j in range(len(Ortho)))
        norm_new = np.sqrt(new_vec.T @ (A @ new_vec))

        if norm_new < Tol:
            if verbose:
                print(f"Vector {i} discarded (norm {norm_new} below threshold)")
            # Even discarded, still pad the projection to match columns
            padded_proj = np.zeros(len(Ortho))
            padded_proj[:len(new_proj)] = new_proj
            Coeff.append(padded_proj)
            continue

        new_vec /= norm_new

        if Adjust:
            while np.any(np.abs([q.T @ (A @ new_vec) for q in Ortho]) > Tol):
                cur_proj = np.array([q.T @ (A @ new_vec) for q in Ortho])
                new_vec -= sum(Ortho[j] * cur_proj[j] for j in range(len(Ortho)))
                cur_norm = np.sqrt(new_vec.T @ (A @ new_vec))
                new_vec /= cur_norm

        Ortho.append(new_vec)
        new_proj = np.append(new_proj, norm_new)

        # Pad previous Coeff columns if needed
        Coeff = [np.pad(col, (0, 1), 'constant') for col in Coeff]
        Coeff.append(new_proj)

        SubIdx.append(i)

    Ortho = np.column_stack(Ortho)
    Coeff = np.column_stack(Coeff)

    return Ortho, Coeff, SubIdx



def plot_scalar_map(ax, mesh, scalar_field, cmap='coolwarm'):
    from matplotlib import colors

    # Normalize scalar field for consistent colormap mapping
    norm = colors.Normalize(vmin=-np.max(np.abs(scalar_field)), vmax=np.max(np.abs(scalar_field)))

    # Let plot_trisurf handle scalar_field itself by passing as `array`
    trisurf = ax.plot_trisurf(mesh.vertices[:, 0],
                              mesh.vertices[:, 1],
                              mesh.vertices[:, 2],
                              triangles=mesh.faces,
                              cmap=cmap,
                              array=scalar_field,
                              linewidth=0,
                              edgecolor='none',
                              antialiased=True,
                              shade=True)

    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=0, azim=90)

    return trisurf

# def plot_scalar_map_pyvista(plotter, mesh, scalar_field, row, col, title, cmap='bwr'):
#     pv_mesh = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).astype(np.int32))
#     pv_mesh.point_data['scalar_field'] = scalar_field
#
#     plotter.subplot(row, col)
#     plotter.add_mesh(pv_mesh, scalars='scalar_field', cmap=cmap, show_edges=False, smooth_shading=True)
#     plotter.add_text(title, font_size=FontSize)
#     plotter.view_xy()
# def plot_scalar_map_pyvista(plotter, mesh, scalar_field, row, col, title, clim=(-1, 1), cmap='bwr'):
#     # Normalize safely and remove NaNs/Infs
#     scalar_field = np.nan_to_num(scalar_field)
#     if np.max(np.abs(scalar_field)) > 0:
#         scalar_field = scalar_field / np.max(np.abs(scalar_field))
#
#     pv_faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).astype(np.int32)
#     pv_mesh = pv.PolyData(mesh.vertices, pv_faces)
#     pv_mesh.point_data['scalar_field'] = scalar_field
#
#     plotter.subplot(row, col)
#     plotter.add_mesh(pv_mesh,
#                      scalars='scalar_field',
#                      cmap=cmap,
#                      show_edges=False,
#                      lighting=False,               # Disable lighting completely (optional: try ambient=1)
#                      smooth_shading=False,         # Force flat shading like Matlab
#                      clim=clim,
#                      show_scalar_bar=False)        # We add one global colorbar instead
#
#     plotter.add_text(title, font_size=FontSize)
#     plotter.view_xy()
def plot_scalar_map_pyvista(plotter, mesh, scalar_field, row, col, title, clim=(-1, 1), cmap='coolwarm'):
    # Normalize safely
    scalar_field = np.clip(np.nan_to_num(scalar_field), -1, 1)
    if np.max(np.abs(scalar_field)) > 0:
        scalar_field = scalar_field / np.max(np.abs(scalar_field))


    # actor = plotter.add_text(f"φ{Idx}", font_size=FontSize)
    # actor.GetTextProperty().SetFontFamilyToArial()
    # Create PV mesh
    pv_faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).astype(np.int32)
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)
    pv_mesh.point_data['scalar_field'] = scalar_field
    # Select subplot
    plotter.subplot(row, col)

    # Plot with only the scalar overlay, no visible mesh body (like Matlab)
    plotter.add_mesh(pv_mesh,
                     scalars='scalar_field',
                     cmap=cmap,
                     clim=clim,
                     show_edges=False,
                     interpolate_before_map=True,
                     show_scalar_bar=True,
                     smooth_shading=True,
                     lighting=True,
                     ambient=0.3,
                     diffuse=1.0,
                     specular=0.3,
                     opacity=1.0,
                     nan_opacity=0.0)  # Ensure any NaN is fully invisible

    # Text & camera
    plotter.add_text(title, font_size=FontSize)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.5)




# Usage per subplot:
use_pyvista = True  # Set to True to use PyVista for plotting

# -------------------------
# Load mesh and compute eigenfunctions
# -------------------------
print("Loading mesh and computing eigenfunctions... ")
mesh = trimesh.load_mesh(MeshFile, file_type='off')
vertices = mesh.vertices
faces = mesh.faces

# Assuming you have your FEM operator functions
# from your_utils.mesh_processing import compute_fem_matrices, compute_eigenfunctions
# from your_utils.orthogonalization import compute_eigenproducts, orthogonalize_basis
# from your_utils.plotting import plot_scalar_map

NumFuncs = NumRows * NumCols

# FEM stiffness and mass matrices (with Dirichlet BC assumed)
S, A = compute_fem_matrices(vertices, faces, order=1, bc='Dirichlet')

# Eigen decomposition (shift-invert for smallest eigenvalues)
# Lambda, Phi = eigsh(S, M=NumFuncs, sigma=-1e-5, which='LM', OPinv=A)
Lambda, Phi = eigsh(S, k=NumFuncs, M=A, sigma=-1e-5, which='LM')

Lambda = np.real(Lambda)
Phi = np.real(Phi)

# -------------------------
# Compute eigenproducts
# -------------------------
print("Computing eigenproducts... ")
# PolyPhi = compute_eigenproducts(Phi,order=2, include_linear=True)
PolyPhi = compute_eigenproducts(Phi,A, order=2, normalized=False)

# -------------------------
# Orthogonalization
# -------------------------
print("Orthogonalization... ")
# Q, R = orthogonalize_basis(A, PolyPhi, thresh=OrthoThresh, verbose=True)
Q, Coeff, SubIdx = orthogonalize_basis(A, PolyPhi, Tol=OrthoThresh, Adjust=True, verbose=True)
# -------------------------
# Plotting
# -------------------------


if use_pyvista:
    plotter = pv.Plotter(shape=(3 * NumRows, NumCols), window_size=(1600, 900), off_screen=False)

    # Eigenfunctions
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            plot_scalar_map_pyvista(plotter, mesh, Phi[:, Idx], i, j, f"phi{Idx}", cmap='bwr')


    # Eigenproducts
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            Idx1 = Idx // NumFuncs + 1
            Idx2 = (Idx % NumFuncs) + 2 * (Idx1 - 1)
            plot_scalar_map_pyvista(plotter, mesh, PolyPhi[:, NumFuncs + Idx], NumRows + i, j,
                                    f"phi{Idx // NumFuncs + 1}phi{(Idx % NumFuncs) + 2 * (Idx // NumFuncs)}",
                                    cmap='bwr')

    # Orthogonalized eigenproducts
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            plot_scalar_map_pyvista(plotter, mesh, Q[:, NumFuncs + Idx], 2 * NumRows + i, j, f"Q{Idx}", cmap='bwr')

    # Add a global colorbar
    plotter.add_scalar_bar(title="Scalar Field", n_labels=5, vertical=True)
    plotter.show()

else:
    fig = plt.figure(figsize=(20, 10))
    TotalPlots = 3 * NumRows * NumCols

    # Plot eigenfunctions
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            ax = fig.add_subplot(3 * NumRows, NumCols, Idx + 1, projection='3d')
            plot_scalar_map(ax, mesh, Phi[:, Idx], cmap='bwr')
            ax.set_title(f"$\\varphi_{{{Idx}}}$", fontsize=FontSize)

    # Plot eigenproducts
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            ax = fig.add_subplot(3 * NumRows, NumCols, NumRows * NumCols + Idx + 1, projection='3d')
            plot_scalar_map(ax, mesh, PolyPhi[:, NumFuncs + Idx], cmap='bwr')
            Idx1 = Idx // NumFuncs + 1
            Idx2 = (Idx % NumFuncs) + 2 * (Idx1 - 1)
            ax.set_title(f"$\\varphi_{{{Idx1}}}\\varphi_{{{Idx2}}}$", fontsize=FontSize)

    # Plot orthogonalized eigenproducts
    for i in range(NumRows):
        for j in range(NumCols):
            Idx = i * NumCols + j
            ax = fig.add_subplot(3 * NumRows, NumCols, 2 * NumRows * NumCols + Idx + 1, projection='3d')
            plot_scalar_map(ax, mesh, Q[:, NumFuncs + Idx], cmap='bwr')
            ax.set_title(f"$Q_{{{Idx}}}$", fontsize=FontSize)
    plt.tight_layout()
    plt.show()