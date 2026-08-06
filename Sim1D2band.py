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
    r_tsp = params.get("r_tsp", 0.0)
    t_tsp = params.get("t_tsp", 0.0)
    r_tps = params.get("r_tps", 0.0)
    t_tps = params.get("t_tps", 0.0)
    e_s = params.get("e_s", 0.0)
    e_p = params.get("e_p", 0.0)
    ks = 200
    a = 1.0

    write_to_output(f'r_tsp: {r_tsp}; t_tsp: {t_tsp}; r_tps: {r_tps}; t_tps: {t_tps}')
    t_sp = r_tsp * math.e**(-1j*t_tsp)
    t_ps = r_tps * math.e**(-1j*t_tps)
    write_to_output(f't_sp: {t_sp}')
    write_to_output(f't_ps: {t_ps}')

    k_werte = np.linspace(-np.pi/a, np.pi/a, ks)

    ew_oben = []
    ew_unten = []
    bandlücke_k = []
    V_k_abs = []
    H_diffs = []
    evs = []

    # Vorzeichen umgedreht, im Vergleich zu Formeln, damit die Parabeln anders herum sind.
    for k in k_werte:
        H_00 = e_s + 2 * t_ss * math.cos(k*a)
        H_01 = t_ps * math.e**(-1j*k*a) - t_sp * math.e**(1j*k*a)
        H_10 = t_sp * math.e**(-1j*k*a) - t_ps * math.e**(1j*k*a)
        H_11 = e_p + 2 * t_pp * math.cos(k*a)

        H_k = np.array([[H_00, H_01], [H_10, H_11]])
        E, ev = np.linalg.eigh(H_k)
        ew_unten.append(E[0])
        ew_oben.append(E[1])
        evs.append(ev)
        bandlücke_k.append(E[1] - E[0])
        V_abs01 = abs(H_01) #2 * abs(t_sp) * abs(math.sin(k * a))
        #V_abs10 = abs(H_10)
        V_k_abs.append(V_abs01)
        H_diff = abs(H_00 - H_11)
        H_diffs.append(H_diff)


    alle_energien = np.concatenate([ew_oben, ew_unten])
    #E_Fermi = np.max(ew_unten)#np.median(alle_energien)
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

    # Referenz-Bänder ohne Hybridisierung (t_sp = 0, t_ps = 0)
    ew_oben_ref = []
    ew_unten_ref = []

    for k in k_werte:
        H_00_ref = e_s + 2 * t_ss * math.cos(k*a)
        H_11_ref = e_p + 2 * t_pp * math.cos(k*a)
        H_k_ref = np.array([[H_00_ref, 0], [0, H_11_ref]])
        E_ref = np.linalg.eigvalsh(H_k_ref)
        ew_unten_ref.append(E_ref[0])
        ew_oben_ref.append(E_ref[1])

    fig = ax.figure
    # Plotten der beiden Bänder
    ax.plot(k_werte, ew_oben, label='(p)-band', color='red')
    ax.plot(k_werte, ew_unten, label='(s)-band', color='blue')
    ax.plot(k_werte, V_k_abs, label=r'$|V(k)|$', color='#42994B')
    #ax.plot(k_werte, H_diffs, label=r'$|H_{00}-H_{11}|$', color='#4FCD1C')

    #ax.plot(k_werte, s_sa, color='black', linestyle='--')
    #ax.plot(k_werte, s_pa, color='black', linestyle='--')
    #ax.plot(k_werte, p_sa, color='yellow', linestyle=':')
    #ax.plot(k_werte, p_pa, color='yellow', linestyle=':')
    #write_to_output("Eigenvectors: ")
    #write_to_output("s s-Anteil: black --")
    #write_to_output("s p-Anteil: black --")
    #write_to_output("p s-Anteil: yellow :")
    #write_to_output("p p-Anteil: yellow :")
    #ax.axhline(y=E_Fermi, color='#A52BFB', linestyle='-.', linewidth=1.2, label='Fermi-Energy')
    #ax.axvline(x=math.pi, color='gray', linestyle='-', linewidth=1.0)
    #ax.axvline(x=-math.pi, color='gray', linestyle='-', linewidth=1.0)
    #ax.axhline(y=e_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)
    #ax.axhline(y=e_p, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)

    ax.plot(k_werte, ew_oben_ref, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.plot(k_werte, ew_unten_ref, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=0.0, color='gray', linestyle='-', linewidth=1.0)

    if t_sp != 0:
        ax.fill_between(k_werte, ew_unten, ew_oben, color='gray', alpha=0.2, label='band gap')
        #ax.vlines(x=k_min, ymin=ew_unten[idx_min], ymax=ew_oben[idx_min], color='black', linestyle='--', linewidth=1)
        #ax.vlines(x=-k_min, ymin=ew_unten[idx_min], ymax=ew_oben[idx_min], color='black', linestyle='--', linewidth=1)
        write_to_output(f"Minimale Bandlücke: {bandlücke_min:.6f} bei k = {k_min:.4f}")


    # Achsenbeschriftungen und Legende
    ax.set_xlabel(r'$k$ (wave number)', fontsize=12)
    ax.set_ylabel(r'$E(k)$', fontsize=12)
    #ax.lengend()
    ax.legend(loc='center left', bbox_to_anchor=(1.045, 0.5))

    param_text = (
        f"t_ss = {t_ss:.2f}\n"
        f"t_pp = {t_pp:.2f}\n"
        f"t_sp: r={r_tsp:.2f}   θ:{t_tsp:.2f} \n"
        f"t_ps: r={r_tps:.2f}   θ:{t_tps:.2f} \n"
        f"e_s = {e_s:.2f}\n"
        f"e_p = {e_p:.2f}"
    )

    for t in fig.texts:
        t.remove()
    #        0.75, 0.79
    fig.text(0.95, 0.14, param_text, fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.savefig("plot.png", dpi=400, bbox_inches="tight")

    #write_to_output(f"Fermi-Energie: {E_Fermi:.6f}")
    # if t_sp == 0:
    #     write_to_output("--> Metall")
    # elif E_Fermi < np.min(ew_oben):
    #     if bandlücke_min < 3.0:
    #         write_to_output("--> Halbleiter")
    #     elif bandlücke_min >= 3.0:
    #         write_to_output("--> Isolator")
    # elif E_Fermi >= np.min(ew_oben):
    #     write_to_output("trotz t_sp != 0 --> Metall")


