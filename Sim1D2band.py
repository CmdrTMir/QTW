import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math


def twoband_1D(ax, write_to_output, params):
    # ---- Variablen ----
    t_ss = params.get("t_ss", 1.0)
    t_pp = params.get("t_pp", 1.0)
    t_sp = params.get("t_sp", 0.0)
    e_s = params.get("e_s", 1.0)
    e_p = params.get("e_p", 1.0)
    ks = 200
    a = 1.0

    k_werte = np.linspace(-np.pi/a, np.pi/a, ks)

    ew_oben = []
    ew_unten = []
    #####
    ##### # plus = nach unten offene Parabel
    ##### # minus = nach oben offene Parabel
    ##### oder man definiert t < 0 ?
    #####
    for k in k_werte:
        H_00 = e_s + 2 * t_ss * math.cos(k*a)
        H_10 = 2j * t_sp * math.sin(k*a)
        H_01 = -2j * t_sp * math.sin(k*a)
        H_11 = e_p + 2 * t_pp * math.cos(k*a)

        H_k = np.array([[H_00, H_01], [H_10, H_11]])
        E = np.linalg.eigvalsh(H_k)
        ew_unten.append(E[0])
        ew_oben.append(E[1])



    # Plotten der beiden Bänder
    ax.plot(k_werte, ew_oben, label='oberes Band (p)', color='red')
    ax.plot(k_werte, ew_unten, label='unteres Band (s)', color='blue')
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axhline(y=e_s, color='orange', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.axhline(y=e_p, color='orange', linestyle='-', linewidth=0.8, alpha=0.7)

    # Achsenbeschriftungen und Legende
    ax.set_xlabel(r'$k$ (Wellenzahl)')
    ax.set_ylabel(r'$E(k)$ (Energie)')
    ax.legend()
    #ax.grid(True, linestyle='--', alpha=0.6)

