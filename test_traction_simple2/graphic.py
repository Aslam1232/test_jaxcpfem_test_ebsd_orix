#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# 1) Chemin vers les logs
base_dir = os.path.dirname(__file__)
np_dir   = os.path.join(base_dir, 'data', 'numpy', 'traction_cube_debug')

# 2) Chargement des données
eps_g, sp_avg, svm_avg = np.loadtxt(os.path.join(np_dir,'global_poly.txt'),
                                     unpack=True, skiprows=1)
eps_l, sp_loc, svm_loc = np.loadtxt(os.path.join(np_dir,'local_poly.txt'),
                                     unpack=True, skiprows=1)
dist = np.loadtxt(os.path.join(np_dir,'dist_poly.txt'), skiprows=1)
eps_d = dist[:,0]
poly_min, poly_q25, poly_med, poly_q75, poly_max = dist[:,1:].T
eps_s, slip_max = np.loadtxt(os.path.join(np_dir,'slip_poly.txt'),
                             unpack=True, skiprows=1)

# 3) Tracé des graphiques
fig, axes = plt.subplots(2,2, figsize=(12,10))

# a) Contrainte moyenne
axes[0,0].plot(eps_g, sp_avg, label='σzz moyen')
axes[0,0].plot(eps_g, svm_avg,'--', label='σ_VM moyen')
axes[0,0].set(title='σ_moyen vs εzz', xlabel='εzz', ylabel='Contrainte (MPa)')
axes[0,0].legend(); axes[0,0].grid()

# b) Comportement de la maille 0
axes[0,1].plot(eps_l, sp_loc, label='σzz cellule 0')
axes[0,1].plot(eps_l, svm_loc,'--', label='σ_VM cellule 0')
axes[0,1].set(title='σ locale vs εzz', xlabel='εzz', ylabel='Contrainte (MPa)')
axes[0,1].legend(); axes[0,1].grid()

# c) Distribution des contraintes (quartiles)
axes[1,0].fill_between(eps_d, poly_q25, poly_q75, alpha=0.3, label='IQR σzz')
axes[1,0].plot(eps_d, poly_med,    label='Médiane σzz')
axes[1,0].set(title='Distr. σzz (Q1–Q3)', xlabel='εzz', ylabel='σzz (MPa)')
axes[1,0].legend(); axes[1,0].grid()

# d) Glissement plastique maximal
axes[1,1].plot(eps_s, slip_max, label='max γ_inc')
axes[1,1].set(title='Glissement max γ_inc vs εzz',
              xlabel='εzz', ylabel='γ_inc')
axes[1,1].legend(); axes[1,1].grid()

plt.tight_layout()
plt.show()
