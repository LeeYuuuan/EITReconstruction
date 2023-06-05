
from __future__ import absolute_import, division, print_function
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.mesh.shape import thorax
from pyeit.mesh.wrapper import PyEITAnomaly_Circle



"""set premeters"""
h0 = 0.08


""" 0. build mesh """
n_el = 16  # nb of electrodes
mesh_obj = mesh.create(n_el, h0=h0)

""" 1. problem setup """
anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=10.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
mesh_new.perm[0:10] = 10
print(mesh_new.perm.shape)

protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# calculate simulated data
fwd = EITForward(mesh_obj, protocol_obj)
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm)

""" 3. naive inverse solver using back-projection """
eit = bp.BP(mesh_obj, protocol_obj)
eit.setup(weight="none")
# the normalize for BP when dist_exc>4 should always be True
ds = 192.0 * eit.solve(v1, v0, normalize=True)

# extract node, element, alpha
pts = mesh_obj.node
tri = mesh_obj.element


fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(12, 9))
# original
ax = axes[0,0]
ax.axis("equal")
print(ax.axis)
ax.set_title(r"Input $\Delta$ Conductivities")
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)

im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat", edgecolors="black")
# im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(mesh_new.perm), shading="flat", edgecolors="black")
# print(mesh_new.perm)
# reconstructed
ax1 = axes[1,0]
im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds, edgecolors="black")
ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
ax1.axis("equal")
fig.colorbar(im, ax=axes.ravel().tolist())
# fig.savefig('../doc/images/demo_bp.png', dpi=96)
print(f"The number of triangle:{len(tri)}")
plt.show()