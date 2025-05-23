import sys
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
import numpy as np
import trimesh
from PyQt5 import QtWidgets
from pyvistaqt import QtInteractor
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from os import path

matplotlib.use('Qt5Agg')



# -------------------------
# FEM matrices
# -------------------------
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


# -------------------------
# Eigenproducts
# -------------------------
def compute_eigenproducts(Phi, A, order=2, normalized=False):
    n_vertices, n_funcs = Phi.shape
    PolyPhi = [Phi[:, 0]]
    Basis = Phi[:, 1:]
    all_indices = []
    for degree in range(1, order + 1):
        all_indices.extend(combinations_with_replacement(range(Basis.shape[1]), degree))
    for idx_combo in all_indices:
        product = np.ones(n_vertices)
        for idx in idx_combo:
            product *= Basis[:, idx]
        PolyPhi.append(product)
    PolyPhi = np.column_stack(PolyPhi)
    if normalized:
        norms = np.sqrt(np.einsum('ij,ij->j', PolyPhi, A @ PolyPhi))
        PolyPhi = PolyPhi / norms
    return PolyPhi


# -------------------------
# Orthogonalization
# -------------------------
def orthogonalize_basis(A, Basis, Tol=1e-9):
    Ortho = []
    Coeff = []
    q0 = Basis[:, 0] / np.sqrt(Basis[:, 0].T @ (A @ Basis[:, 0]))
    Ortho.append(q0)
    Coeff.append(np.array([np.sqrt(Basis[:, 0].T @ (A @ Basis[:, 0]))]))
    for i in range(1, Basis.shape[1]):
        new_proj = np.array([q.T @ (A @ Basis[:, i]) for q in Ortho])
        new_vec = Basis[:, i] - sum(Ortho[j] * new_proj[j] for j in range(len(Ortho)))
        norm_new = np.sqrt(new_vec.T @ (A @ new_vec))
        if norm_new < Tol:
            continue
        new_vec /= norm_new
        Ortho.append(new_vec)
        new_proj = np.append(new_proj, norm_new)
        Coeff = [np.pad(col, (0, 1)) for col in Coeff]
        Coeff.append(new_proj)
    Ortho = np.column_stack(Ortho)
    Coeff = np.column_stack(Coeff)
    return Ortho, Coeff


# -------------------------
# Cp of C
# -------------------------
def cp_of_c(C):
    return np.kron(C, C)


# -------------------------
# Mesh and eigenproblem
# -------------------------
Src_mesh = trimesh.load_mesh(path.join(path.dirname(__file__), "../Meshes/TOSCA/wolf0.off"), file_type='off')
Trg_mesh = trimesh.load_mesh(path.join(path.dirname(__file__), "../Meshes/TOSCA/wolf2.off"), file_type='off')

S_src, A_src = compute_fem_matrices(Src_mesh.vertices, Src_mesh.faces)
S_trg, A_trg = compute_fem_matrices(Trg_mesh.vertices, Trg_mesh.faces)

K = 4
N = 2

eigvals_src, Phi_src = eigsh(S_src, k=K, M=A_src, sigma=-1e-5)
eigvals_trg, Phi_trg = eigsh(S_trg, k=K, M=A_trg, sigma=-1e-5)

PolyPhi_src = compute_eigenproducts(Phi_src, A_src, order=N)
PolyPhi_trg = compute_eigenproducts(Phi_trg, A_trg, order=N)

Q_src, R_src = orthogonalize_basis(A_src, PolyPhi_src)
Q_trg, R_trg = orthogonalize_basis(A_trg, PolyPhi_trg)

C = Phi_trg.T @ (A_trg @ Phi_src)
CTilde = cp_of_c(C)

O = np.zeros((Q_trg.shape[1], Q_src.shape[1]))
O[:K, :K] = C
U, _, _, _ = np.linalg.lstsq(Q_trg, PolyPhi_src, rcond=None)

for idx in range(K, Q_src.shape[1]):
    O[:, idx] = (1 / R_src[idx, idx]) * (U[:, idx] - O[:, :idx] @ R_trg[:idx, idx])
    O[:K, idx] = 0

Mtx = Q_trg.T @ (A_trg @ Q_src)


# -------------------------
# Qt Application
# -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined Mesh and Matrices Viewer")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QGridLayout(central_widget)

        # PyVista widgets
        self.pv_widget_src = QtInteractor()
        self.pv_widget_trg = QtInteractor()

        # Add source mesh
        self.pv_widget_src.add_mesh(Src_mesh, color='white', show_edges=False)
        self.pv_widget_src.view_xy()
        self.pv_widget_src.reset_camera()

        # Add target mesh
        self.pv_widget_trg.add_mesh(Trg_mesh, color='white', show_edges=False)
        self.pv_widget_trg.view_xy()
        self.pv_widget_trg.reset_camera()

        # Matplotlib widgets
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(O, cmap='bwr')
        ax1.set_title('Analytic')
        plt.colorbar(im1, ax=ax1)

        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(Mtx, cmap='bwr')
        ax2.set_title('Ground Truth')
        plt.colorbar(im2, ax=ax2)

        canvas1 = FigureCanvas(fig1)
        canvas2 = FigureCanvas(fig2)

        # Layout
        layout.addWidget(self.pv_widget_src.interactor, 0, 0)
        layout.addWidget(self.pv_widget_trg.interactor, 0, 1)
        layout.addWidget(canvas1, 1, 0)
        layout.addWidget(canvas2, 1, 1)


# Run app
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()
