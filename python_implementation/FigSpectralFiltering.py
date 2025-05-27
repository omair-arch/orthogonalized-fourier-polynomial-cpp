import random
from os import path
import trimesh
import numpy as np
import common_functions as com
import pyvista as pv
from scipy.sparse import dia_matrix
import scipy.sparse.linalg

# Reproducibility
seednum = 0
random.seed(seednum)


# Parameters
SrcFile = '../Meshes/sheep.off'
KSrc = 15
N = 3          # Please, keep this 2 or 3
OrthoThresh = 1e-2
OrthoThreshReiter = 1e-9
SVDThresh = (KSrc - 2) * 2e-3
FontSize = 14
PlotTitles = True
tau = 1e-2

# Load the meshes and compute the needed fields
print("Loading the meshes... ")
Src = trimesh.load_mesh(path.join(path.dirname(__file__), SrcFile), file_type='off')
Src.vertices = Src.vertices / np.sqrt(np.sum(com.tri_areas(Src)))

print("Solve the eigenproblem... ")
stiffness, _, mass = com.FEM_higher(Src, 1, 'Dirichlet')
lam, phi = scipy.sparse.linalg.eigs(stiffness, M=mass, k=N * KSrc, sigma=-1e-5)
lam = scipy.sparse.dia_matrix((lam, 0), shape = (len(lam),len(lam)))

# Initialize a function to transfer and plot it
# f = Src.vertices # sin(16 .* pi .* Src.X .* Src.Y)

# Transfer with eigenproducts
polyPhi = com.eigprods(Src, KSrc - 1, S=stiffness, A=mass, phi=phi, lam=lam, n=N, normalized=True)
freqs = com.compute_dirichlet(stiffness, polyPhi)
indices = np.argsort(freqs)
freqs = freqs[indices]
basis_poly = polyPhi
basis_poly = basis_poly[:,indices]
del indices, phi, lam
# filter = exp(-0.01.*(abs(-freqs(1)+freqs)))';
filter = np.exp(-tau*(np.abs(np.arange(0, np.size(basis_poly, 1))))).transpose()
filter = filter / np.max(filter)

# coeff = pinv(basis_poly)*f;
coeff = np.linalg.pinv(np.sqrt(mass) @ basis_poly) @ np.sqrt(mass) * Src.vertices
prod = basis_poly @ (filter * coeff)

# Transfer with stressed orthogonal basis
QS = com.orthogonalize_basis(mass, polyPhi, tol=OrthoThreshReiter, adjust=True)
freqs = com.compute_dirichlet(stiffness, QS)
indices = np.argsort(freqs)
freqs = freqs[indices]
basis_our = QS
basis_our = basis_our[:,indices]
del indices, stiffness, QS, freqs
# filter = exp(-0.01.*(abs(-freqs(1)+freqs)))';
# filter = exp(-tau*(abs(0:size(basis_our,2)-1)))';
# filter = filter./max(filter);
# figure; plot(filter)

coeff = basis_our.transpose() @ (mass @ Src.vertices)
ortho_forced = basis_our @ (filter * coeff)

#
New = Src
PlotMesh = Src
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
pl = pv.Plotter(shape=(2, 4))
pl.subplot((0,0))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Source", font_size=FontSize)  #view([0 90])

New.vertices = prod
pl.subplot((0,1))
PlotMesh = New
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Poly", font_size=FontSize) #view([0 90])

New.vertices = ortho_forced
pl.subplot((0,2))
PlotMesh = New
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Otho", font_size=FontSize) #view([0 90])

pl.subplot((0,3))
chart = pv.Chart2D()
chart.plot(filter)
pl.add_chart(chart)
Filter1 = filter

# Transfer with eigenproducts
filter = np.exp(-0.0025*((np.abs(basis_poly.shape[1]-1)- np.arange(0, basis_poly.shape[1])))).transpose()
# filter = ones(size(basis_poly,2),1)+rand(size(basis_poly,2),1)*0.5; % 
# filter = exp(-tau*(abs(0:size(basis,2)-1)))';
# filter = filter./max(filter);

# coeff = pinv(basis)*f;
coeff = np.pinv(np.sqrt(mass) @ basis_poly) @ np.sqrt(mass) @ Src.vertices
prod = basis_poly @ (filter * coeff)

# Transfer with stressed orthogonal basis

coeff = basis_our.transpose() @ (mass @ Src.vertices)
ortho_forced = basis_our @ (filter * coeff)

del basis_poly, basis_our, coeff, mass
#
New = Src
New.vertices = prod
pl.subplot((1,0))
PlotMesh = Src
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Source", font_size=FontSize)  #view([0 90])

pl.subplot((1,1))
PlotMesh = New
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Poly", font_size=FontSize)  #view([0 90])

New.vertices = ortho_forced
pl.subplot((1,2))
PlotMesh = New
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*170, [1, 0, 0]))
PlotMesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/180*25, [0, 1, 0]))
mesh = pv.PolyData(PlotMesh.vertices, PlotMesh.faces)
mesh["f"] = np.zeros(Src.vertices.shape[0], 1)
pl.add_mesh(mesh, scalars="f", cmap="viridis",specular=0.1, diffuse=0.35, show_edges=False)
pl.add_title("Otho", font_size=FontSize)  #view([0 90])

pl.subplot((1,3))
chart = pv.Chart2D()
chart.plot(filter)
pl.add_chart(chart)
Filter2 = filter
pl.show()

pl2 = pv.Plotter(shape = (2,1))
pl2.subplot((0,0))
chart = pv.Chart2D()
chart.plot(Filter1)
pl2.add_chart(chart)

pl2.subplot((1,0))
chart = pv.Chart2D()
chart.plot(Filter2)
pl2.add_chart(chart)
pl2.show()








































