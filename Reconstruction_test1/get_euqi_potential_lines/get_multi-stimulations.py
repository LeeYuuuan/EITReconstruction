from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import os
"""set peremeters"""

ex_type = "ad" # current path
number_of_line = 64 # the number of lines

# the number of target is 1
target_number = 2

if target_number == 1:
    target_cond = 10.0


# the number of target is 2
if target_number == 2:
    number_of_line = 64
    target_cond = [50, 100]


"""make dictionary"""
raw_path = 'D:/Users/PC/OneDrive/EIT/ModelAnalysis/equi-potential lines/multi_stimulations'
corr_path = raw_path + f'/target_number={target_number},number_of_line={number_of_line},cond={target_cond}/'
# print(corr_path)

if os.path.exists(corr_path):
    print(True)
else:
    os.mkdir(corr_path)


"""0. build mesh"""
n_el = 16  # nb of electrodes
for ex in range(n_el):
    mesh_obj = mesh.create(n_el, h0=0.08)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos
    x, y = pts[:, 0], pts[:, 1]
    mesh_obj.print_stats()


    # change permittivity
    if target_number == 1:
        anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.2, perm=target_cond)
    if target_number == 2:
        anomaly = [
    PyEITAnomaly_Circle(center=[0.4, 0], r=0.2, perm=target_cond[0]),
    PyEITAnomaly_Circle(center=[-0.4, 0], r=0.2, perm=target_cond[1])
        ]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new.perm

    """ 1. FEM forward simulations """
    # setup (AB) current path
    if ex_type == "op":
        ex_line = [ex, (ex+n_el/2)%16]
    if ex_type =="ad":
        ex_line = [ex, (ex+1)%16]

    # calculate simulated data using FEM
    fwd = Forward(mesh_new)
    f = fwd.solve(ex_line)
    f = np.real(f)
    # print(f.shape)

    """ 2. plot """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # draw equi-potential lines
    vf = np.linspace(min(f), max(f), 64)
    # print(vf)

    ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)
    # draw mesh structure
    ax1.tripcolor(
        x,
        y,
        tri,
        np.real(perm),
        edgecolors="k",
        shading="flat",
        alpha=0.5,
        cmap=plt.cm.Greys,
    )
    # draw electrodes
    ax1.plot(x[el_pos], y[el_pos], "ro")
    for i, e in enumerate(el_pos):
        ax1.text(x[e], y[e], str(i + 1), size=12)
    ax1.set_title("equi-potential lines")
    # clean up
    ax1.set_aspect("equal")
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    fig.set_size_inches(5, 5)
    fig.subplots_adjust(top=0.975, bottom=0.02, left=0.15)
    fig_path = corr_path + f'ex[{ex_line[0]},{ex_line[1]}].png'

    print(fig_path)
    
    fig.savefig(fig_path)
# plt.show()



