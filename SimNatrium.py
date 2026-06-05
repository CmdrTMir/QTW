import numpy as np
import matplotlib.pyplot as plt

def Natrium(ax):
    # ============================================================
    # PARAMETER für 1D-Natrium-Kette (Näherung)
    # ============================================================
    a = 3.7                      # Gitterkonstante von Natrium (in Angström, ca.)
    t = -0.5                     # Hopping-Parameter in eV (typisch für Na)

    Nk = 200                     # Anzahl der k-Punkte

    # ============================================================
    # k-Werte: symmetrisch von -pi/a bis +pi/a
    # ============================================================
    k_vals = np.linspace(-np.pi/a, np.pi/a, Nk, endpoint=False)

    # Energie-Dispersion für ein einfaches s-Band (nächste Nachbarn)
    # ε(k) = -2t cos(k*a)  (das chemische Potential mu verschiebt nur, wir lassen es hier weg)
    E_band = -2 * t * np.cos(k_vals * a)

    # ============================================================
    # PLOT
    # ============================================================
    #plt.figure(figsize=(10, 6))

    ax.plot(k_vals, E_band, 'b-', linewidth=2, label='s-Band (3s-Valenzelektronen)')

    # Fermi-Energie berechnen (für halbe Füllung)
    # Bei halber Füllung (ein Elektron pro Platz) liegt die Fermi-Energie bei E=0 (wegen Symmetrie der Cosinus-Dispersion)
    # Das Maximum des Bandes ist +2t, das Minimum -2t. Bei halber Füllung ist EF=0.
    ax.axhline(y=0.0, color='r', linestyle='--', linewidth=1.5, label='Fermi-Energie (E_F) – halbe Füllung')

    # Brillouin-Zonen-Grenzen und Γ-Punkt
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, label='Γ-Punkt (k=0)')
    ax.axvline(x=np.pi/a, color='gray', linestyle='--', linewidth=0.8, label='Brillouin-Zonen-Rand (π/a)')
    ax.axvline(x=-np.pi/a, color='gray', linestyle='--', linewidth=0.8)

    # Achsen
    ax.set_xlabel('k (1/Å)', fontsize=12)
    ax.set_ylabel('Energie E (eV)', fontsize=12)
    ax.set_title('1D-Natrium-Kette (Tight-Binding, ein Band) – Metall', fontsize=14)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    #plt.show()

# ============================================================
# AUSGABE
# ============================================================
#print("\n" + "="*50)
#print("ERGEBNISSE für die 1D-Natrium-Kette:")
#print("="*50)
#print(f"Gitterkonstante a        : {a:.2f} Å")
#print(f"Hopping-Parameter t      : {t:.2f} eV")
#print(f"Bandbreite               : {4*t:.2f} eV")
#print(f"Bandminimum bei k=0      : {-2*t:.2f} eV")
#print(f"Bandmaximum bei k=±π/a   : {2*t:.2f} eV")
#print(f"Fermi-Energie (halbe Füllung): 0.00 eV (liegt innerhalb des Bandes)")
#print("="*50)
