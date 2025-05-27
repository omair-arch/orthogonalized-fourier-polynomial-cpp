import numpy as np
import scipy.sparse as sp
import os
import pickle
from sympy import symbols, diff, integrate, Matrix
from scipy.sparse import dia_matrix, csr_matrix
import scipy.sparse.linalg
import trimesh

def FEM_higher(M, p, bc='Neumann', ICache='Cache/FEM/'):
    """
    Higher-order FEM discretization of Laplace-Beltrami operator

    Parameters:
        M: Mesh object with VERT, TRIV, n, m attributes
        p: Order of FEM discretization (1 for linear)
        bc: Boundary conditions ('Neumann' or 'Dirichlet')
        ICache: Directory for caching basis function matrices

    Returns:
        S: Stiffness matrix (sparse CSR)
        A: Mass matrix (sparse CSR)
        Al: Lumped mass matrix (sparse CSR)
    """
    # Input validation
    bc = bc.lower()
    if bc not in ['neumann', 'dirichlet']:
        raise ValueError("Boundary condition must be 'Neumann' or 'Dirichlet'")

    # Special case for linear FEM
    if p == 1:
        return cotangent_method(M, bc)

    # Get basis function matrices
    I1, I2, I3, I4 = calc_FEM_test_funcs(p, ICache)
    q = I1.shape[0]  # Number of basis functions per element

    # Compute adjacency matrix for higher-order nodes
    Ad = calc_indices_adj_order_p(M, p)

    # Compute total number of nodes
    Nedges = np.count_nonzero(np.triu(Ad.toarray()))
    OnEdges = p - 1
    Internal = (p + 1) * (p + 2) // 2 - (3 * p)
    Ntot = M.n + OnEdges * Nedges + Internal * M.m

    # Build element-to-node mapping
    TRItot = build_element_node_map(M, p, Ad, Ntot, OnEdges, Nedges)

    # Compute local matrices
    Alocal, Blocal = compute_local_matrices(M, I1, I2, I3, I4)

    # Assemble global matrices
    S, A = assemble_global_matrices(M, TRItot, Alocal, Blocal, Ntot, q, bc)

    # Lumped mass matrix
    Al = sp.diags(A.sum(axis=1).flatten(), 0, shape=(Ntot, Ntot))

    if bc == 'neumann' and Al.nnz < Ntot:
        print(f"Warning: Lumped mass matrix is singular ({Al.nnz}/{Ntot} non-zeros)")

    return S.tocsr(), A.tocsr(), Al.tocsr()


def cotangent_method(M, bc):
    """Linear FEM using cotangent weights"""
    angles = np.zeros((M.faces.shape[0], 3))

    for i in range(3):
        a = i % 3
        b = (i + 1) % 3
        c = (i + 2) % 3

        ab = M.vertices[M.faces[:, b]] - M.vertices[M.faces[:, a]]
        ac = M.vertices[M.faces[:, c]] - M.vertices[M.faces[:, a]]

        cos_theta = np.sum(ab * ac, axis=1) / (np.linalg.norm(ab, axis=1) * np.linalg.norm(ac, axis=1))
        angles[:, a] = cos_theta / np.sqrt(1 - cos_theta ** 2 + 1e-10)

    # Stiffness matrix assembly
    ii = np.concatenate([M.faces[:, 0], M.faces[:, 1], M.faces[:, 2],
                         M.faces[:, 2], M.faces[:, 1], M.faces[:, 0]])
    jj = np.concatenate([M.faces[:, 1], M.faces[:, 2], M.faces[:, 0],
                         M.faces[:, 1], M.faces[:, 0], M.faces[:, 2]])
    vals = np.concatenate([angles[:, 2], angles[:, 0], angles[:, 1],
                           angles[:, 0], angles[:, 2], angles[:, 1]]) * 0.5

    W = sp.coo_matrix((-vals, (ii, jj)), shape=(M.vertices.shape[0], M.vertices.shape[0])).tocsc()

    W.setdiag(-np.array(W.sum(axis=1)).flatten())



    # Mass matrix assembly
    areas = tri_areas(M)
    vals = np.tile(areas / 12, 6)
    Sc = sp.coo_matrix((vals, (ii, jj)), shape=(M.vertices.shape[0], M.vertices.shape[0])).tocsc()
    Sc.setdiag(Sc.sum(axis=1).A.flatten())

    # Apply boundary conditions
    if bc == 'dirichlet':
        # Find boundary edges
        E = np.vstack([M.faces[:, [0, 1]], M.faces[:, [1, 2]], M.faces[:, [2, 0]]])
        E = np.sort(E, axis=1)
        Eu, Euidx = np.unique(E, axis=0, return_index=True)
        E = np.delete(E, Euidx, axis=0)
        Boundary = Eu[~np.isin(Eu, E).all(axis=1)].flatten()

        Inner = np.setdiff1d(np.arange(M.vertices.shape[0]), Boundary)

        W_full = W.copy()
        W = sp.dok_matrix((M.vertices.shape[0], M.vertices.shape[0]))
        W[Boundary, Boundary] = 1
        W[Inner[:, None], Inner] = W_full[Inner[:, None], Inner]

        Sc_full = Sc.copy()
        Sc = sp.dok_matrix((M.vertices.shape[0], M.vertices.shape[0]))
        Sc[Inner[:, None], Inner] = Sc_full[Inner[:, None], Inner]

    # Lumped mass matrix
    Sl = sp.diags(Sc.sum(axis=1).A.flatten(), 0)

    return (W + W.T) * 0.5, (Sc + Sc.T) * 0.5, Sl


def calc_indices_adj_order_p(M, p):
    """Compute adjacency matrix for higher-order nodes"""
    ii = np.concatenate([M.TRIV[:, 0], M.TRIV[:, 1], M.TRIV[:, 2],
                         M.TRIV[:, 2], M.TRIV[:, 1], M.TRIV[:, 0]])
    jj = np.concatenate([M.TRIV[:, 1], M.TRIV[:, 2], M.TRIV[:, 0],
                         M.TRIV[:, 1], M.TRIV[:, 0], M.TRIV[:, 2]])

    Ad = sp.coo_matrix((np.ones(6 * M.m), (ii, jj)), shape=(M.n, M.n)).tocsc()
    Ad = (Ad + Ad.T) > 0

    # Get upper triangular part and assign increasing values
    Ad_upper = sp.triu(Ad, k=1)
    ii, jj = Ad_upper.nonzero()
    values = np.arange(1, (p - 1) * len(ii) + 1, p - 1)

    return sp.coo_matrix((values, (ii, jj)), shape=(M.n, M.n)).tocsc()


def build_element_node_map(M, p, Ad, Ntot, OnEdges, Nedges):
    """Create mapping from elements to all nodes (vertices + edge + internal nodes)"""
    q = (p + 1) * (p + 2) // 2  # Total nodes per element
    TRItot = np.zeros((M.m, q), dtype=int)
    TRItot[:, :3] = M.TRIV  # Vertex nodes

    # Edge nodes
    for ii in range(1, OnEdges + 1):
        i = ii + 2  # 0-based indexing

        # Edge 0-1
        edge_idx = np.ravel_multi_index((M.TRIV[:, 0], M.TRIV[:, 1]), Ad.shape)
        TRItot[:, i] = Ad.data[edge_idx] + M.n + \
                       (OnEdges - ii) * (M.TRIV[:, 0] > M.TRIV[:, 1]) + \
                       (ii - 1) * (M.TRIV[:, 0] <= M.TRIV[:, 1])

        # Edge 1-2
        i += OnEdges
        edge_idx = np.ravel_multi_index((M.TRIV[:, 1], M.TRIV[:, 2]), Ad.shape)
        TRItot[:, i] = Ad.data[edge_idx] + M.n + \
                       (OnEdges - ii) * (M.TRIV[:, 1] > M.TRIV[:, 2]) + \
                       (ii - 1) * (M.TRIV[:, 1] <= M.TRIV[:, 2])

        # Edge 2-0
        i += OnEdges
        edge_idx = np.ravel_multi_index((M.TRIV[:, 2], M.TRIV[:, 0]), Ad.shape)
        TRItot[:, i] = Ad.data[edge_idx] + M.n + \
                       (OnEdges - ii) * (M.TRIV[:, 2] > M.TRIV[:, 0]) + \
                       (ii - 1) * (M.TRIV[:, 2] <= M.TRIV[:, 0])

    # Internal nodes
    Internal = q - 3 - 3 * OnEdges
    for i in range(Internal):
        TRItot[:, q - Internal + i] = np.arange(M.m) + Ntot - Internal * M.m + i * M.m

    return TRItot


def compute_local_matrices(M, I1, I2, I3, I4):
    """Compute local stiffness and mass matrices for each element"""
    P1 = M.VERT[M.TRIV[:, 1]] - M.VERT[M.TRIV[:, 0]]
    P2 = M.VERT[M.TRIV[:, 2]] - M.VERT[M.TRIV[:, 0]]

    P11 = np.sum(P1 * P1, axis=1)
    P22 = np.sum(P2 * P2, axis=1)
    P12 = np.sum(P1 * P2, axis=1)

    pre = np.linalg.norm(np.cross(P1, P2), axis=1)

    # Local stiffness matrices
    Alocal = (P11[:, None, None] * I2 +
              P22[:, None, None] * I1 -
              P12[:, None, None] * I3) / pre[:, None, None]

    # Local mass matrices
    Blocal = I4 * pre[:, None, None]

    return Alocal, Blocal


def assemble_global_matrices(M, TRItot, Alocal, Blocal, Ntot, q, bc):
    """Assemble global stiffness and mass matrices"""
    # Global indices
    ii = TRItot[:, np.tile(np.arange(q), q)].flatten()
    jj = TRItot[:, np.repeat(np.arange(q), q)].flatten()

    # Find boundary nodes (vertices on boundary edges)
    E = np.vstack([M.TRIV[:, [0, 1]], M.TRIV[:, [1, 2]], M.TRIV[:, [2, 0]]])
    E = np.sort(E, axis=1)
    Eu, Euidx = np.unique(E, axis=0, return_index=True)
    E = np.delete(E, Euidx, axis=0)
    BoundaryVerts = Eu[~np.isin(Eu, E).all(axis=1)].flatten()

    # Stiffness matrix
    if bc == 'neumann':
        S = sp.coo_matrix((Alocal.flatten(), (ii, jj)), shape=(Ntot, Ntot))
    else:
        S_full = sp.coo_matrix((Alocal.flatten(), (ii, jj)), shape=(Ntot, Ntot))
        Inner = np.setdiff1d(np.arange(Ntot), BoundaryVerts)
        S = sp.dok_matrix((Ntot, Ntot))
        S[BoundaryVerts, BoundaryVerts] = 1
        S[Inner[:, None], Inner] = S_full[Inner[:, None], Inner]

    # Mass matrix
    if bc == 'neumann':
        A = sp.coo_matrix((Blocal.flatten(), (ii, jj)), shape=(Ntot, Ntot))
    else:
        A_full = sp.coo_matrix((Blocal.flatten(), (ii, jj)), shape=(Ntot, Ntot))
        A = sp.dok_matrix((Ntot, Ntot))
        A[Inner[:, None], Inner] = A_full[Inner[:, None], Inner]

    return S.tocsr(), A.tocsr()


def calc_FEM_test_funcs(Order, ICache):
    """Compute or load cached basis function matrices"""
    cache_file = f"{ICache}/Order{Order}.pkl"

    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Compute symbolically if not cached
    u, v = symbols('u v')
    U = [u ** 0]
    V = [v ** 0]

    # Basis polynomials
    for i in range(1, Order + 1):
        U.append(u * U[-1])
        V.append(v * V[-1])

    UV = Matrix(V).T * Matrix(U)
    UV = np.fliplr(np.array(UV).astype(np.float64))

    # Get all basis functions
    basis = []
    for i in range(Order + 1):
        basis.extend(np.diag(UV, Order - i))
    basis = Matrix(basis)

    # Vandermonde matrix for interpolation
    M = []
    points = [(0, 0), (1, 0), (0, 1)]  # Vertex points

    # Edge points
    Space = np.linspace(0, 1, Order + 1)
    for i in range(1, Order):
        points.append((Space[i], 0))
        points.append((1 - Space[i], Space[i]))
        points.append((0, 1 - Space[i]))

    # Internal points
    for i in range(1, Order):
        for j in range(1, Order - i + 1):
            points.append((Space[i], Space[j]))

    # Build Vandermonde matrix
    M = Matrix([[basis[k].subs({u: p[0], v: p[1]}) for k in range(len(basis))]
                for p in points])

    # Compute dual basis coefficients
    C = np.linalg.inv(np.array(M).astype(np.float64))

    # Compute basis function matrices
    size = len(basis)
    I1 = np.zeros((size, size))
    I2 = np.zeros((size, size))
    I3 = np.zeros((size, size))
    I4 = np.zeros((size, size))

    H = (Matrix(C.T) * basis).expand()

    for i in range(size):
        for j in range(i, size):
            # Mass matrix terms
            f = H[i] * H[j]
            I4[i, j] = float(integrate(integrate(f, (v, 0, 1 - u)), (u, 0, 1)))

            # Stiffness matrix terms
            f = diff(H[i], u) * diff(H[j], u)
            I1[i, j] = float(integrate(integrate(f, (v, 0, 1 - u)), (u, 0, 1)))

            f = diff(H[i], v) * diff(H[j], v)
            I2[i, j] = float(integrate(integrate(f, (v, 0, 1 - u)), (u, 0, 1)))

            f = diff(H[i], v) * diff(H[j], u) + diff(H[i], u) * diff(H[j], v)
            I3[i, j] = float(integrate(integrate(f, (v, 0, 1 - u)), (u, 0, 1)))

    # Symmetrize matrices
    I1 = I1 + I1.T - np.diag(np.diag(I1))
    I2 = I2 + I2.T - np.diag(np.diag(I2))
    I3 = I3 + I3.T - np.diag(np.diag(I3))
    I4 = I4 + I4.T - np.diag(np.diag(I4))

    # Save to cache
    os.makedirs(ICache, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((I1, I2, I3, I4), f)

    return I1, I2, I3, I4


def tri_areas(M):
    """Compute triangle areas"""
    v1 = M.vertices[M.faces[:, 0]]
    v2 = M.vertices[M.faces[:, 1]]
    v3 = M.vertices[M.faces[:, 2]]

    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
def orthogonalize_basis(A, Basis, tol=1e-9, adjust=True, return_coeff=False, return_subidx=False):
    """
    Orthogonalize a basis using Gram-Schmidt with area/mass-weighted inner product (using matrix A).
    Equivalent to the provided MATLAB orthogonalize function.

    Parameters:
    -----------
    A : ndarray (n, n)
        Area or mass matrix.
    Basis : ndarray (n, m)
        Input basis to orthogonalize (n vertices, m functions).
    tol : float, optional
        Tolerance for discarding near-zero vectors.
    adjust : bool, optional
        Whether to adjust repeatedly if the vector is still not orthogonal.
    return_coeff : bool, optional
        If True, also return the coefficient matrix.
    return_subidx : bool, optional
        If True, also return the surviving indices.

    Returns:
    --------
    Ortho : ndarray (n, k)
        Orthogonalized basis.
    Coeff : ndarray (optional)
        Coefficient matrix such that Ortho @ Coeff ≈ Basis.
    SubIdx : list (optional)
        Indices of surviving basis vectors.
    """

    Ortho = []
    Coeff = []
    SubIdx = []

    q0 = Basis[:, 0]
    norm0 = np.sqrt(q0.T @ (A @ q0))
    q0 /= norm0
    Ortho.append(q0)
    if return_coeff:
        Coeff.append(np.array([norm0]))
    if return_subidx:
        SubIdx.append(0)

    for i in range(1, Basis.shape[1]):
        NewProj = np.array([vec.T @ (A @ Basis[:, i]) for vec in Ortho])
        New = Basis[:, i] - Ortho @ NewProj
        Norm = np.sqrt(New.T @ (A @ New))

        if Norm < tol:
            if return_coeff:
                padded_proj = np.zeros(len(Ortho))
                padded_proj[:len(NewProj)] = NewProj
                Coeff.append(padded_proj)
            continue

        New /= Norm

        if adjust:
            while np.any(np.abs([vec.T @ (A @ New) for vec in Ortho]) > tol):
                CurProj = np.array([vec.T @ (A @ New) for vec in Ortho])
                New -= Ortho @ CurProj
                CurNorm = np.sqrt(New.T @ (A @ New))
                New /= CurNorm

        Ortho.append(New)
        NewProj = np.append(NewProj, Norm)
        if return_coeff:
            # Pad existing Coeff columns
            Coeff = [np.pad(c, (0, 1)) for c in Coeff]
            Coeff.append(NewProj)
        if return_subidx:
            SubIdx.append(i)

    Ortho = np.column_stack(Ortho)
    if return_coeff:
        Coeff = np.column_stack(Coeff)
        if return_subidx:
            return Ortho, Coeff, SubIdx
        else:
            return Ortho, Coeff
    else:
        if return_subidx:
            return Ortho, SubIdx
        else:
            return Ortho
import numpy as np
from scipy.sparse.linalg import eigsh

def sub_index(order, k):
    """
    Equivalent of the MATLAB sub_index function to get unique indices.

    Returns:
    --------
    subs_order : ndarray
        Indices of unique polynomial combinations.
    """
    PolyPhi = np.arange(k + 1).reshape(1, -1)
    Basis = np.arange(1, k + 1)
    Prods = Basis.reshape(1, -1)

    for i in range(2, order + 1):
        tmp1 = np.tile(Basis, (1, k ** (i - 1)))
        tmp2 = np.reshape(np.tile(Prods, (k, 1)), (i - 1, k ** i))
        Prods = np.vstack((tmp1, tmp2))
        PolyPhi = np.vstack((PolyPhi, np.zeros((1, PolyPhi.shape[1]))))
        PolyPhi = np.hstack((PolyPhi, Prods))

    PolyPhi = np.sort(PolyPhi, axis=0)
    _, unique_indices = np.unique(PolyPhi.T, axis=0, return_index=True)
    subs_order = np.sort(unique_indices)

    return subs_order
import numpy as np

def compute_dirichlet(S, f):
    """
    Compute Dirichlet energy of the given function(s).

    Parameters:
    -----------
    S : scipy.sparse matrix (n x n)
        Stiffness (Laplace-Beltrami) matrix.
    f : ndarray (n, m)
        Function(s) over the mesh vertices. Can be single (n,) or multiple (n, m).

    Returns:
    --------
    energies : ndarray (m,)
        Dirichlet energy per function.
    """
    if f.ndim == 1:
        f = f[:, np.newaxis]

    # Ensure matrix multiplication correctness
    energies = np.einsum('ij,ij->j', f, S @ f)

    return energies

def eigprods(M, k, S=None, A=None, phi=None, lam=None, n=2, normalized=False):
    """
    PolyPhi = EIGPRODS(M, k, S=None, A=None, phi=None, lam=None, n=2, normalized=false) 
    Decide if normalize or not the polynomial basis so that the functions have squared norm
    equals to 1 on the mesh.
    """
    
    # Check for the mass and stiffness matrices
    if A==None or S==None:
        S, A = FEM_higher(M, 1, 'Dirichlet')

    # Check if the basis has already been computed and has enough elements
    if phi==None or phi.shape[1] < k + 1:
        phi, lam = scipy.sparse.linalg.eigs(S, M=A, k=(k+1)*n, sigma=-1e-5)
        lam = dia_matrix((lam, 0), shape = (len(lam),len(lam)))
    
    PolyPhi = phi[:, np.arange(0,k+1)]
    Basis = PolyPhi[:, np.arange(1, PolyPhi.shape[0])]
    Prods = Basis
    for i in range(2, n+1):
        tmp1 = np.matlib.repmat(Basis, 1, k**(i - 1))
        tmp2 = np.matlib.repmat(Prods, k, 1).reshape(A.shape[0], k**i)
        Prods = tmp1 * tmp2
        PolyPhi = np.hstack((PolyPhi, Prods))
    
    PolyPhi = PolyPhi[:, sub_index(n, k)]
    
    
    if normalized:
        norms = np.diag(PolyPhi.T @ A @ PolyPhi)
        PolyPhi = PolyPhi / np.sqrt(norms.T)
    
    return PolyPhi

def sub_index(order, k):

    PolyPhi = np.arange(0, k+1)
    Basis = np.arange(1, k+1)
    Prods = Basis
    for i in range(2, order+1):
        tmp1 = np.matlib.repmat(Basis, k**(i - 1), 1)
        tmp2 = np.matlib.repmat(Prods, 1, k).reshape (i - 1, k**i)
        Prods = np.vstack((tmp1, tmp2))
        PolyPhi = np.vstack(PolyPhi, np.zeros(1, np.max(PolyPhi.shape)))
        PolyPhi = np.hstack((PolyPhi, Prods))
    PolyPhi = np.sort(PolyPhi, axis=0)
    _, subs_order = np.unique(PolyPhi.T, return_index=True, axis=0)
    return subs_order

def utils_cmaps_bwr(Resolution=256):
#cm = BWR(Resolution) Returns the blue-white-red colormap with the desired
#resolution
#
#cm = BWR Returns the blue-white-red colormap with resolution 256
    
    Colors = np.array([[0, 0, 0.8],    # Blue
              [1, 1, 1],    # White
              [0.8, 0, 0]])   # Red
    cm = utils_cmaps_blend(Colors, Resolution)
    return cm

def utils_cmaps_blend(Colors, Resolution=256):
#cmap = BLEND(Colors, Resolution) Generates a colormap with the given
#resolution which passes through all the colors in the given matrix. Notice
#the matrix Colors is n-by-3.
#
#cmap = BLEND(Colors) Uses the default resolution of 256 entries

    
    # Get the number of colors and the number of entries between each color
    NumColors = Colors.shape[0]
    ColorEntries = np.floor(Resolution / (NumColors - 1))
    
    # Initialize the colormap
    if NumColors == 1:
        raise ValueError("Cannot blend one color. Please, use utils.cmaps.constant for this.");
    cmap = np.zeros([Resolution, 3])
    
    # Fill the colormap
    for i in range (1, NumColors):
        # Compute the begin and end indices
        Begin = (i - 1) * ColorEntries
        End = Begin + ColorEntries
        # Compute the channels
        Channels = {}
        Channels['R'] = np.linspace(Colors[i-1, 0], Colors[i, 0], ColorEntries).T
        Channels['G'] = np.linspace(Colors[i-1, 1], Colors[i, 1], ColorEntries).T
        Channels['B'] = np.linspace(Colors[i-1, 2], Colors[i, 2], ColorEntries).T
        # Fill the map
        cmap[np.arange(Begin, End), :] = [Channels['R'], Channels['G'], Channels['B']]
    return cmap

def mesh_transform_normalize(M):
#NORMALIZE Summary of this function goes here
#   Detailed explanation goes here
    M.apply_transform(trimesh.transformations.scale_matrix(1 / np.sqrt(np.sum(tri_areas(M)))))
    return M

def mesh_proc_heat_kernel(M, t, k= None, basis = None, support=None):
#K = HEAT_KERNEL(M, t, k) Computes the approximation of the heat kernel of
#a mesh with k eigenfunctions of the Laplacian.
#
#K = HEAT_KERNEL(M, t) Computes the approximation of the heat kernel of the
#mesh with all the eigenfunctions already computed.
#
#   It computes the approximation of the heat kernel of the mesh M, using
#   the first k eigenfunctions of the laplacian of M. The computed heat
#   kernel is parametric with respect to time instant t.
#   The heat kernel is represented as an n-by-n matrix, where n is the
#   number of vertices of M and each cell (i, j) contains the heat
#   diffusion from the vertex i from vertex j.

    # If the laplace basis has not yet been computed, compute it
    if basis==None:
        if k==None:
            raise ValueError("If the given mesh has not a precomputed " \
                            "basis, then a number of eigenfunctions for " \
                            "the approximation must be given.")
        _, _, support, basis, _, basis_len = mesh_proc_compute_laplace_basis(M, k)
    # If the laplace basis is not large enough, the recompute it
    elif k==None and basis.shape[1] < k:
        evecs, evals, support, basis, phi0, basis_len = mesh_proc_compute_laplace_basis(M, k)
    # Otherwise, a basis is provided, but not approximation limit is given.
    # In this case, use all of the available basis
    else:
        k = basis.shape[1]
    # Compute the heat kernel
    K = basis[:, np.arange(0, k)] @ np.diag(np.exp(-support * t)) @ basis[:, np.arange(0, k)].T
    return K

def mesh_proc_compute_laplace_basis(M, k, A=None, S=None, basis=None):
#M = compute_laplace_basis(M, k) Computes the first k eigenfunctions of the
#Laplace-Beltrami basis for the given mesh. Notice that the first constant
#eigenfunction is excluded from the basis.
#
#   This is a full list of the newly computed fields of the structure:
#       - A:            The lumped mass matrix. If present, it is not
#                       recomputed.
#       - S:            The stiffness matrix. If present, it is not
#                       recomputed.
#       - evecs:        The first k + 1 eigenfuntions of the
#                       Laplace-Beltrami basis, including the constant one.
#       - evals:        The eigenvalues associated to 'evecs'.
#       - basis:        The first k eigenfunctions of the Laplace-Beltrami 
#                       basis, excluding the constant one.
#       - support:      The eigenvalues associated to 'basis'.
#       - phi0:         The value assumed by the constant eigenfunction in
#                       the whole mesh.
#       - basis_len:    The length of the basis. Namely, k.

    if A==None or S==None:
        S, A, _ = mesh_proc_laplacian(M)
    if basis!=None:
        if basis.shape[1] >= k:
            return [],[]
    if k >= M.vertices.shape[0]:
        print(f"The number of eigenfunctions {k} exceeds rank of the laplacian {M.vertices.shape[0] - 1}.")
        k = M.vertices.shape[0] - 1
    evecs, evals = scipy.sparse.linalg.eigs(S, M=A, k=k+1, sigma=-1e-5)
    evals = np.diag(evals)
    support = evals[np.arange(1,k+1)]
    basis = evecs[:, np.arange(1,k+1)]
    phi0 = evecs[0,0]
    basis_len = k
    return evecs, evals, support, basis, phi0, basis_len

def mesh_proc_laplacian(M, order=1):
#[S, A] = LAPLACIAN(M)
#Computes the stifness and mass matrix for the Laplace-Beltrami operator,
#using Finite Element discretization.
#
#[S, A, Al] = LAPLACIAN(M)
#Computes the stifness and mass matrix for the Laplace-Beltrami operator,
#using Finite Element discretization. Furthermore, the function returns the
#lumped mass matrix, which is the diagonal of the mass.
#
#[__] = LAPLACIAN(M, order)
#The Finite Element dicretization is computed using hat functions of the
#given order.
    
    # Parameters are ok?
    if order != 1 and order != 2 and order != 3:
        raise ValueError(f"The order must be integer in [1, 3]. {order} given.")
    
    if order == 1:
        W, Sc, Sl = mesh_proc_calc_LB_FEM(M)
    elif order == 2:
        W, Sc, Sl = mesh_proc_calc_LB_FEM_quad(M)
    else:
        W, Sc, Sl = mesh_proc_calc_LB_FEM_cubic(M)
    
    return W, Sc, Sl

def mesh_proc_calc_LB_FEM(M):
    # Stiffness (p.s.d.)
    angles = np.zeros(M.faces.shape[0],3)
    for i in range (1,4):
        a = (i-1) % 3
        b = i%3
        c = (i+1) % 3
        ab = M.vertices[M.TRIV[:,b],:] - M.vertices[M.faces[:,a],:]
        ac = M.vertices[M.TRIV[:,c],:] - M.vertices[M.faces[:,a],:]
        #normalize edges
        ab = ab / (np.sqrt(np.sum(np.pow(ab,2), axis=1)) @ np.array([1, 1, 1]))
        ac = ac / (np.sqrt(np.sum(np.pow(ac,2), axis=1)) @ np.array([1, 1, 1]))
        # normalize the vectors
        # compute cotan of angles
        angles[:,a] = np.cot(np.acos(np.sum(ab*ac, axis=1)))
        #cotan can also be computed by x/sqrt(1-x^2)

    indicesI = np.vstack([M.faces[:,0], M.faces[:,1], M.faces[:,2], M.faces[:,2], M.faces[:,1], M.faces[:,0]]).flatten()
    indicesJ = np.vstack([M.faces[:,1], M.faces[:,2], M.faces[:,0], M.faces[:,1], M.faces[:,0], M.faces[:,2]]).flatten()
    values   = np.vstack([angles[:,2], angles[:,0], angles[:,1], angles[:,0], angles[:,2], angles[:,2]]).flatten() * 0.5
    
    W = csr_matrix((-values, (indicesI, indicesJ)),shape=[M.vertices.shape[0], M.vertices.shape[0]])
    W = W - csr_matrix(np.sum(W), (np.arange(0, M.vertices.shape[0]), np.arange(0, M.vertices.shape[0])))

    # Mass
    areas = tri_areas(M)
    values = np.hstack([areas.flatten(), areas.flatten(), areas.flatten(),
                        areas.flatten(), areas.flatten(), areas.flatten(),])/12
    Sc = csr_matrix((values, (indicesI, indicesJ)),shape=[M.vertices.shape[0], M.vertices.shape[0]])
    Sc = Sc+csr_matrix(np.sum(Sc), (np.arange(0, M.vertices.shape[0]), np.arange(0, M.vertices.shape[0])))
    
    # Lumped mass
    Sl = dia_matrix((sum(Sc,2), 0), shape=[M.vertices.shape[0], M.vertices.shape[0]]);
    
    boundary = mesh_proc_calc_boundary_edges(M.faces)
    boundary = np.unique(boundary.flatten())
#     E = [M.TRIV(:, [1 2]);
#          M.TRIV(:, [2 3]);
#          M.TRIV(:, [3 1])];
#     M.Adj = sparse(E(:, 1), E(:, 2), 1, M.n, M.n);
#     M.Adj = double(logical((M.Adj + M.Adj') > 0));
#     bbound = zeros(M.n, 1);
#     bbound(boundary) = 1;
#     bbound = logical(M.Adj * bbound);
#     bbound = find(bbound);
#     [W, Sc, Sl] = dirichlet_bc(W, Sc, Sl, [boundary; bbound], M.n);
    W, Sc, Sl = mesh_proc_dirichlet_bc(W, Sc, Sl, boundary, M.n);
    
    W = (W + W.T)/2
    Sc = (Sc + Sc.T)/2
    return W, Sc, Sl

def mesh_proc_dirichlet_bc(W_full, A_full, AL_full, boundary, n):
    boundary = np.unique(boundary.flatten())
    l = boundary.shape[0]
    inside = np.setdiff1d(np.arange(0,n), boundary)

    W = csr_matrix((n, n))
    W[boundary, boundary] = W_full[boundary, boundary] # eye(l)
    W[inside, inside] = W_full[inside, inside]
    A = csr_matrix((n, n))
    A[inside,inside] = A_full[inside, inside]
    AL = csr_matrix((n, n))
    AL[inside,inside] = AL_full[inside, inside]
    return W, A, AL

def mesh_proc_calc_boundary_edges(triangles):
    if type(triangles) == trimesh.base.Trimesh:
        faces = triangles.faces
    else:
        faces = triangles

    c,d,_,_ = get_boundary(faces)
    bd = np.zeros([np.max(d.shape),2])

    for i in range(0,np.max(c.shape)):
        t = faces[c[i],:]
        v = np.array([True, True, True])
        v[d[i]] = False
        v = t[v]
        bd[i,0] = v[0]
        bd[i,1] = v[1]
    return bd

def get_boundary( tri ):
#GET_BOUNDARY determines the boundary edges of a triangular mesh 
#   [c,d] = get_boundary(tri) takes as input a list tri of consistently oriented triangles
#   returns the indices c of the triangles the boundary edges belong to and
#   the (local) indices d (in {1,2,3}) of the vertices opposing the boundary
#   edge
#   One gets the global indices of those vertices via F(sub2ind(size(F),c,d))
#   Via 
#   d1 = mod(d+1,3); d1(d1==0) = 3;
#   d2 = mod(d+2,3); d2(d2==0) = 3;
#   one gets the local indices of the boundary vertices.


    if tri.shape[0] < tri.shape[1]:
        tri=tri.T
    m = tri.shape[0];

    # Check for opposing halfedges

    # Matrix of directed edges
    I = np.array([tri[:,0], tri[:,1]],
        [tri[:,1], tri[:,2]],
        [tri[:,2], tri[:,0]])
    b = isMemberRows(I[:,[1,0]], I)
    b = np.argwhere(b==False);

    # Triangle indices
    c = np.mod(b,m)
    c[c==0] = m

    # vertex opposing boundary edge
    d = np.floor((b-1)/m)
    d[d==0]=3

    # Directed boundary edges
    I=I[b,:]

    # Boundary vertices
    v = I[I!=0];
    v = np.unique(v)

    return c, d, I, v

def calc_LB_FEM_cubic(M):

    Ia = 1/13440 * np.array(
        [[76, 11, 11, 18, 0, 27, 27, 0, 18, 36],
        [11, 76, 11, 0, 18, 18, 0, 27, 27, 36],
        [11, 11, 76, 27, 27, 0, 18, 18, 0, 36],
        [18, 0, 27, 540, -189 -135, -54, -135, 270, 162],
        [0, 18, 27, -189, 540, 270, -135, -54, -135, 162],
        [27, 18, 0, -135, 270, 540, -189, -135, -54, 162],
        [27, 0, 18, -54, -135, -189, 540, 270, -135, 162],
        [0, 27, 18, -135, -54, -135, 270, 540, -189, 162],
        [18, 27, 0, 270, -135, -54, -135, -189, 540, 162],
        [36, 36, 36, 162, 162, 162, 162, 162, 162, 1944]])

    Ib = 1/80 * np.array(
        [[34, -7, 0, -54, 27, -3, -3, 3, 3, 0],
        [-7, 34, 0, 27, -54, 3, 3, -3, -3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-54, 27, 0, 135, -108, 0, 0, 0, 0, 0],
        [27, -54, 0, -108, 135, 0, 0, 0, 0, 0],
        [-3, 3, 0, 0, 0, 135, -27, 27, 27, -162],
        [-3, 3, 0, 0, 0, -27, 135, -135, 27, 0],
        [3, -3, 0, 0, 0, 27, -135, 135, -27, 0],
        [3, -3, 0, 0, 0, 27, 27, -27, 135, -162],
        [0, 0, 0, 0, 0, -162, 0, 0 -162, 324]])

    Ic = 1/80 * np.array(
        [[34, 0, -7, 3, 3, -3, -3, 27, -54, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-7, 0, 34, -3, -3, 3, 3, -54, 27, 0],
        [3, 0, -3, 135, -27, 27, 27, 0, 0, -162],
        [3, 0, -3, -27, 135, -135, 27, 0, 0, 0],
        [-3, 0, 3, 27, -135, 135, -27, 0, 0, 0],
        [-3, 0, 3, 27, 27, -27, 135, 0, 0, -162],
        [27, 0, -54, 0, 0, 0, 0, 135, -108, 0],
        [-54, 0, 27, 0, 0, 0, 0, -108, 135, 0],
        [0, 0, 0, -162, 0, 0, -162, 0, 0, 324]])

    Id = 1/80 * np.array(
        [[68, -7, -7, -51, 30, -6, -6, 30, -51, 0],
        [-7, 0, 7, 24 -57, 57, -24, 0, 0, 0],
        [-7, 7, 0, 0, 0, -24, 57, -57, 24, 0],
        [-51, 24, 0, 135, -108, 27, 27, -27, 135 -162],
        [30, -57, 0, -108, 135, -135, 27, -27, -27, 162],
        [-6, 57, -24, 27, -135, 135, 54, 27, 27, -162],
        [-6, -24, 57, 27, 27, 54, 135, -135, 27, -162],
        [30, 0, -57, -27 -27, 27, -135, 135, -108, 162],
        [-51, 0, 24, 135, -27, 27, 27, -108, 135, -162],
        [0, 0, 0, -162, 162, -162, -162, 162, -162, 324]])

    q = Ib.shape[0]

    Ad = mesh_proc_calc_indices_adj_cubic(M)
    Nedges = np.count_nonzero(np.triu(Ad))
    Ntot = M.vertices.shape[0] + 2 * Nedges + M.faces.shape[0]

    TRIE = np.zeros([M.faces.shape[0], 3])
    TRIE[:, 0] = Ad[np.unravel_index(Ad.shape, M.faces[:, 0], M.faces[:, 1])] + M.vertices.shape[0] + (M.faces[:, 0] > M.faces[:, 1])
    TRIE[:, 1] = Ad[np.unravel_index(Ad.shape, M.faces[:, 0], M.faces[:, 1])] + M.vertices.shape[0] + (not (M.TRIV[:, 0] > M.TRIV[:, 1]))
    TRIE[:, 2] = Ad[np.unravel_index(Ad.shape, M.faces[:, 1], M.faces[:, 2])] + M.vertices.shape[0] + (M.faces[:, 1] > M.faces[:, 2])
    TRIE[:, 3] = Ad[np.unravel_index(Ad.shape, M.faces[:, 1], M.faces[:, 2])] + M.vertices.shape[0] + (not (M.TRIV[:, 1] > M.TRIV[:, 2]))
    TRIE[:, 4] = Ad[np.unravel_index(Ad.shape, M.faces[:, 2], M.faces[:, 0])] + M.vertices.shape[0] + (M.faces[:, 2] > M.faces[:, 0])
    TRIE[:, 5] = Ad[np.unravel_index(Ad.shape, M.faces[:, 2], M.faces[:, 0])] + M.vertices.shape[0] + (not (M.TRIV[:, 2] > M.TRIV[:, 0]))

    TRIE[:, 6] = np.arange(1, M.faces.shape[0]).T + M.vertices.shape[0] + 2*Nedges
    TRItot = np.hstack([M.faces, TRIE])

    P1 = M.vertices[M.faces[:, 1], :] - M.vertices[M.faces[:, 0], :]
    P2 = M.vertices[M.TRIV[:, 2], :] - M.vertices[M.faces[:, 1], :]

    #     Row-wise dot product, repmat to match I#
    p11 = np.tensordot(P1, P1, axes=([1],[1]))
    p11b = np.matlib.repmat(p11, 1, q)
    p11b = p11b.flatten()

    p22 = np.tensordot(P2, P2, axes=([1],[1]))
    p22b = np.matlib.repmat(p22, 1, q)
    p22b = p22b.flatten()

    p12 = np.tensordot(P1, P2, axes=([1],[1]))
    p12b = np.matlib.repmat(p12, 1, q)
    p12b = p12b.flatten()

    pre2 = np.linalg.norm(np.cross(P1, P2, axis=1), axis=1)
    pre2b = np.matlib.repmat(pre2, 1, q)
    pre2b = pre2b.flatten()

    I1b = np.matlib.repmat(Ib, 1, M.m)
    I2b = np.matlib.repmat(Ic, 1, M.m)
    I3b = np.matlib.repmat(Id, 1, M.m)
    I4b = np.matlib.repmat(Ia, 1, M.m)

    Alocal2 = (p11b * I2b + p22b * I1b - p12b * I3b) / pre2b
    Blocal2 = I4b * pre2b

    va = Alocal2.flatten()

    vb = Blocal2.flatten()

    idx_rows = np.matlib.repmat(np.arange(0, M.faces.shape[0]), q*q)
    idx_rows = idx_rows.T.flatten()

    idx_cols = np.repmat(np.arange(0, q), 1, q)
    idx_cols = idx_cols.T.flatten()
    idx_cols = np.matlib.repmat(idx_cols, M.face.shape[0], 1)
    rows = TRItot[np.unravel_index(TRItot.shape, idx_rows, idx_cols)]

    idx_cols = np.matlib.repmat(np.arange(0, q), M.faces.shape[0] * q, 1)
    cols = TRItot(np.unravel_index(TRItot.shape, idx_rows, idx_cols))

    A = csr_matrix((va, (rows, cols)), shape =[Ntot, Ntot])
    B = csr_matrix((vb, (rows, cols)), shape =[Ntot, Ntot])

    Stiff = A
    Mass = B
    # LumpedMass = spdiag(sum(B, 2));
    LumpedMass = dia_matrix((np.sum(B, axis=1), 0), shape = [Ntot, Ntot])
    return Stiff, Mass, LumpedMass

def mesh_proc_calc_indices_adj_cubic(M):
    indicesI = np.vstack([M.faces[:,0], M.faces[:,1], M.faces[:,2], M.faces[:,2], M.faces[:,1], M.faces[:,0]]).flatten()
    indicesJ = np.vstack([M.faces[:,1], M.faces[:,2], M.faces[:,0], M.faces[:,1], M.faces[:,0], M.faces[:,2]]).flatten()
    Ad = csr_matrix((np.ones([M.faces.shape[0], 6]).flatten(), (indicesI, indicesJ)),shape=[M.vertices.shape[0], M.vertices.shape[0]])
    indicesIJ = np.argwhere(np.triu(Ad)!=0)
    indicesI = indicesIJ[:,0]
    indicesJ = indicesIJ[:,1]
    del indicesIJ
    Nedges = np.count_nonzero(np.triu(Ad))
    Ad = csr_matrix((np.arange(1, 2*Nedges, step=2),(indicesI, indicesJ)), shape=[M.vertices.shape[0], M.vertices.shape[0]])
    Ad = Ad + Ad.T
    return Ad

def isMemberRows(A, B):
    M = []
    for i in range (0, A.shape[0]):
        isMember = False
        for j in range(0, B.shape[0]):
            if A[i,:] == B[j,:]:
                isMember = True
                break
        M.append(isMember)
    return np.array(M)

def waveKernelSignature(laplaceBasis, eigenvalues, Ae, numTimes):
# This method computes the wave kernel signature for each vertex on a list.
# It uses precomputed LB eigenstuff stored in "mesh" and automatically
# chooses the time steps based on mesh geometry.

    numEigenfunctions = eigenvalues.shape[0];

    D = laplaceBasis @ (Ae @ np.pow(laplaceBasis,2))

    absoluteEigenvalues = np.abs(eigenvalues);
    emin = np.log(absoluteEigenvalues.T.flatten()[1])
    emax = np.log(absoluteEigenvalues[absoluteEigenvalues.shape - 1])
    s = 7*(emax-emin) / numTimes; # Why 7?
    emin = emin + 2*s
    emax = emax - 2*s
    es = np.linspace(emin,emax,numTimes)

    T = np.exp(-np.pow((np.matlib.repmat(np.log(absoluteEigenvalues),numTimes,1) - 
                 np.matlib.repmat(es,1,numEigenfunctions)),2)/(2*(s**2)))
    wks = D @ T
    wks = laplaceBasis @ wks
    return wks

def waveKernelMap(laplaceBasis, eigenvalues, Ae, numTimes, landmarks):
# This method computes the wave kernel signature for each vertex on a list.
# It uses precomputed LB eigenstuff stored in "mesh" and automatically
# chooses the time steps based on mesh geometry.

    wkms = [];
    #
    for li in range(0, np.max(landmarks.shape)):
        segment = np.zeros([laplaceBasis.shape[0],1])
        segment[landmarks[li]] = 1
        
        numEigenfunctions = eigenvalues.shape[0]
    
        absoluteEigenvalues = np.abs(eigenvalues)
        emin = np.log(absoluteEigenvalues.T.flatten()[1])
        emax = log(absoluteEigenvalues[absoluteEigenvalues.shape - 1])
        s = 7*(emax-emin) / numTimes; # Why 7?
        emin = emin + 2*s
        emax = emax - 2*s
        es = np.linspace(emin,emax,numTimes)
        
        T = np.exp(-np.pow((np.matlib.repmat(np.log(absoluteEigenvalues),numTimes,1) -
                np.matlib.repmat(es,1,numEigenfunctions)),2)/(2*(s**2)))
        wkm = T*np.matlib.repmat(laplaceBasis * segment, 1, size(T,2)).T
        wkm  = laplaceBasis @ wkm;
        
        wkms = np.hstack([wkms, wkm])
    return wkms