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
    bandlücke_k = []
    V_k_abs = []
    evs = []
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
        E, ev = np.linalg.eigh(H_k)
        ew_unten.append(E[0])
        ew_oben.append(E[1])
        evs.append(ev)
        bandlücke_k.append(E[1] - E[0])
        V_abs = 2 * abs(t_sp) * abs(math.sin(k * a)) #np.real(2j * t_sp * math.sin(k*a))
        V_k_abs.append(V_abs)

    alle_energien = np.concatenate([ew_oben, ew_unten])
    E_Fermi = np.max(ew_unten)#np.median(alle_energien)
    bandlücke_min = np.min(np.array(ew_oben) - np.array(ew_unten))
    idx_min = np.argmin(bandlücke_k)
    k_min = k_werte[idx_min]

    s_sa = []
    p_sa = []
    s_pa = []
    p_pa = []
    for vs in evs:#k-mal
        s_sa.append(abs(vs[0][0])**2) # s s-anteil
        p_sa.append(abs(vs[1][0])**2) # p s-anteil
        s_pa.append(abs(vs[0][1])**2) # s p-anteil
        p_pa.append(abs(vs[1][1])**2) # p p-anteil


    # Plotten der beiden Bänder
    ax.plot(k_werte, ew_oben, label='oberes Band (p)', color='red')
    ax.plot(k_werte, ew_unten, label='unteres Band (s)', color='blue')
    ax.plot(k_werte, V_k_abs, label=r'$|V(k)|$', color='#42994B')
    ax.plot(k_werte, s_sa, color='black', linestyle='--')
    ax.plot(k_werte, s_pa, color='black', linestyle='--')
    ax.plot(k_werte, p_sa, color='yellow', linestyle=':')
    ax.plot(k_werte, p_pa, color='yellow', linestyle=':')
    write_to_output("Eigenvectors: ")
    write_to_output("s s-Anteil: black --")
    write_to_output("s p-Anteil: black --")
    write_to_output("p s-Anteil: yellow :")
    write_to_output("p p-Anteil: yellow :")
    ax.axhline(y=E_Fermi, color='#A52BFB', linestyle='--', linewidth=1.2)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=math.pi, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=-math.pi, color='gray', linestyle='-', linewidth=1.0)
    ax.axhline(y=e_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axhline(y=e_p, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)
    if t_sp != 0:
        ax.fill_between(k_werte, ew_unten, ew_oben, color='gray', alpha=0.2, label='Bandlücke')
        ax.vlines(x=k_min, ymin=ew_unten[idx_min], ymax=ew_oben[idx_min], color='black', linestyle='--', linewidth=1)
        ax.vlines(x=-k_min, ymin=ew_unten[idx_min], ymax=ew_oben[idx_min], color='black', linestyle='--', linewidth=1)
        write_to_output(f"Minimale Bandlücke: {bandlücke_min:.6f} bei k = {k_min:.4f}")


    # Achsenbeschriftungen und Legende
    ax.set_xlabel(r'$k$ (Wellenzahl)')
    ax.set_ylabel(r'$E(k)$ (Energie)')
    ax.legend()
    #ax.grid(True, linestyle='--', alpha=0.6)

    write_to_output(f"Fermi-Energie: {E_Fermi:.6f}")
    if t_sp == 0:
        write_to_output("--> Metall")
    elif E_Fermi < np.min(ew_oben):
        if bandlücke_min < 3.0:
            write_to_output("--> Halbleiter")
        elif bandlücke_min >= 3.0:
            write_to_output("--> Isolator")
    elif E_Fermi >= np.min(ew_oben):
        write_to_output("trotz t_sp != 0 --> Metall")


