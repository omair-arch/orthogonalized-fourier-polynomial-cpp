import numpy as np
import scipy.sparse as sp
import os
import pickle
from sympy import symbols, diff, integrate, Matrix
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
    angles = np.zeros((M['m'], 3))

    for i in range(3):
        a = i % 3
        b = (i + 1) % 3
        c = (i + 2) % 3

        ab = M['VERT'][M['TRIV'][:, b]] - M['VERT'][M['TRIV'][:, a]]
        ac = M['VERT'][M['TRIV'][:, c]] - M['VERT'][M['TRIV'][:, a]]

        cos_theta = np.sum(ab * ac, axis=1) / (np.linalg.norm(ab, axis=1) * np.linalg.norm(ac, axis=1))
        angles[:, a] = cos_theta / np.sqrt(1 - cos_theta ** 2 + 1e-10)

    # Stiffness matrix assembly
    ii = np.concatenate([M['TRIV'][:, 0], M['TRIV'][:, 1], M['TRIV'][:, 2],
                         M['TRIV'][:, 2], M['TRIV'][:, 1], M['TRIV'][:, 0]])
    jj = np.concatenate([M['TRIV'][:, 1], M['TRIV'][:, 2], M['TRIV'][:, 0],
                         M['TRIV'][:, 1], M['TRIV'][:, 0], M['TRIV'][:, 2]])
    vals = np.concatenate([angles[:, 2], angles[:, 0], angles[:, 1],
                           angles[:, 0], angles[:, 2], angles[:, 1]]) * 0.5

    W = sp.coo_matrix((-vals, (ii, jj)), shape=(M['n'], M['n'])).tocsc()

    W.setdiag(-np.array(W.sum(axis=1)).flatten())



    # Mass matrix assembly
    areas = tri_areas(M)
    vals = np.tile(areas / 12, 6)
    Sc = sp.coo_matrix((vals, (ii, jj)), shape=(M['n'], M['n'])).tocsc()
    Sc.setdiag(Sc.sum(axis=1).A.flatten())

    # Apply boundary conditions
    if bc == 'dirichlet':
        # Find boundary edges
        E = np.vstack([M['TRIV'][:, [0, 1]], M['TRIV'][:, [1, 2]], M['TRIV'][:, [2, 0]]])
        E = np.sort(E, axis=1)
        Eu, Euidx = np.unique(E, axis=0, return_index=True)
        E = np.delete(E, Euidx, axis=0)
        Boundary = Eu[~np.isin(Eu, E).all(axis=1)].flatten()

        Inner = np.setdiff1d(np.arange(M['n']), Boundary)

        W_full = W.copy()
        W = sp.dok_matrix((M['n'], M['n']))
        W[Boundary, Boundary] = 1
        W[Inner[:, None], Inner] = W_full[Inner[:, None], Inner]

        Sc_full = Sc.copy()
        Sc = sp.dok_matrix((M['n'], M['n']))
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
    v1 = M['VERT'][M['TRIV'][:, 0]]
    v2 = M['VERT'][M['TRIV'][:, 1]]
    v3 = M['VERT'][M['TRIV'][:, 2]]

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

def eigprods(M, k, n=2, normalized=False):
    """
    Compute n-th order polynomial basis from Laplacian eigenfunctions.
    Follows the MATLAB eigprods.m logic exactly.

    Parameters:
    -----------
    M : dict-like
        Must have 'vertices', 'faces', and optionally 'A', 'S', 'Phi', 'Lambda'.
    k : int
        Number of Laplacian eigenfunctions.
    n : int, optional
        Order of polynomial products (default 2).
    normalized : bool, optional
        Whether to normalize basis functions with mass matrix.

    Returns:
    --------
    PolyPhi : ndarray (n_vertices, N)
        The polynomial basis functions.
    """

    # Check mass and stiffness matrices
    if 'A' not in M or 'S' not in M:
        from common_functions import compute_fem_matrices  # Use your common module
        M['S'], M['A'] = compute_fem_matrices(M['vertices'], M['faces'])

    # Compute basis if missing
    if 'Phi' not in M or M['Phi'].shape[1] < k + 1:
        eigvals, Phi = eigsh(M['S'], k=k+1, M=M['A'], sigma=-1e-5)
        M['Phi'] = Phi
        M['Lambda'] = eigvals

    PolyPhi = M['Phi'][:, :k+1]
    Basis = PolyPhi[:, 1:]
    Prods = Basis.copy()

    for i in range(2, n + 1):
        tmp1 = np.tile(Basis, (1, k ** (i - 1)))
        tmp2 = np.reshape(np.tile(Prods, (k, 1)), (M['A'].shape[0], k ** i))
        Prods = tmp1 * tmp2
        PolyPhi = np.hstack((PolyPhi, Prods))

    # Apply sub_index to select unique products (mimicking MATLAB behavior)
    indices = sub_index(n, k)
    PolyPhi = PolyPhi[:, indices]

    if normalized:
        norms = np.sqrt(np.einsum('ij,ij->j', PolyPhi, M['A'] @ PolyPhi))
        PolyPhi = PolyPhi / norms

    return PolyPhi


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
