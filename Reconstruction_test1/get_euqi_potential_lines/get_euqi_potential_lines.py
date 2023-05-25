# coding: utf-8
""" Figure02 for softx """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

"""set peremeters"""
target_perm = np.arange(1,50,0.5)
number_of_line = 64
target_number = 2
default_perm = 10.0 # for multi-target, set the conductivity of one target as const
""" 0. build mesh """

n_el = 16  # nb of electrodes
for tp in target_perm:
    mesh_obj = mesh.create(n_el, h0=0.08)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element
    el_pos = mesh_obj.el_pos
    x, y = pts[:, 0], pts[:, 1]
    mesh_obj.print_stats()


    # change permittivity
    if target_number == 1:
        anomaly = PyEITAnomaly_Circle(center=[0.4, 0.5], r=0.2, perm=target_perm)
    if target_number == 2:
        anomaly = [
    PyEITAnomaly_Circle(center=[0.4, 0], r=0.2, perm=tp),
    PyEITAnomaly_Circle(center=[-0.4, 0], r=0.2, perm=default_perm)
        ]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)
    perm = mesh_new.perm

    """ 1. FEM forward simulations """
    # setup (AB) current path
    i = 6
    ex_line = [i, i+1]

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
    fig.savefig(f'D:/Users/PC/OneDrive/EIT/ModelAnalysis/equi-potential lines/target_number={target_number},number_of_line={number_of_line}/target_cond={tp}.png')
# plt.show()
