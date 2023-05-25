# coding: utf-8
""" demo code for back-projection """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
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
# print("Hello")
""" 0. build mesh """
n_el = 16  # nb of electrodes
v_change = []
gap = 0.1
for cond in np.arange(0, 10, gap):
    
    use_customize_shape = False
    if use_customize_shape:
        # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax
        mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
    else:
        mesh_obj = mesh.create(n_el, h0=0.1)

    """ 1. problem setup """
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.2, perm=cond)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (dist_exc=1), adjacent measures (step_meas=1)
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

    # calculate simulated data
    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)
    v_change.append(v1)

v_count = 3 # the number of electrode for imagining
imgy = []
for j in range(13 * v_count):
    imgy_element = []
    for i in range(20):
        imgy_element.append(v_change[i][j])
    imgy.append(imgy_element)
for j in range(v_count): 
    for i in range(13):
        plt.subplot(4, 4, i + 1)
        plt.plot([i for i in range(20)], imgy[i + j * 13])

for j in range(3):
    plt.subplot(4, 4, 14)
    all_electrode_v = []
    for i in range(13):
        all_electrode_v.append(imgy[i][10 + j * 2])
    plt.plot([i for i in range(13)], all_electrode_v)
print(all_electrode_v)
plt.show()
# print(len(v1))
    
"""
    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element
"""

# draw
# fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))
# original
# ax = axes[0]
# ax.axis("equal")
# ax.set_title(r"Input $\Delta$ Conductivities")
# delta_perm = np.real(mesh_new.perm - mesh_obj.perm)
# im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat", edgecolors="black")
# print(f"The number of triangle:{len(tri)}")


# change conductivity from 0.1 to 1
# print(v0)
# plt.show()
"""print(v0.shape)
plt.imshow(v1.reshape([13, 16]), interpolation='nearest',cmap='bone',origin='upper')
plt.colorbar(shrink=0.5)
plt.xticks(())
plt.yticks(())
plt.show()"""