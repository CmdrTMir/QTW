import numpy as np
import qutip as qt
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
#from qutip import sigmax, sigmaz, destroy, qeye, mesolve, basis

def qutip_lindblad(write_to_output, N_val, t_val, gamma_val):

    # Zeitunabhängige Operatoren
    sigma = qt.destroy(2)
    sig_dag = qt.create(2)
    sigz = qt.sigmaz()
    eye = qt.qeye(2)

    # Parameter
    N = N_val
    t = t_val
    gamma = gamma_val

    w = 1.0
    tlist = np.linspace(0, 1, 10)

    H = 0
    c = []
    c_dag = []

    # ---- berechne H ----
    for j in range(N):
        ops = []
        for k in range(N):
            if k < j:
                ops.append(sigz)
            elif k == j:
                ops.append(sigma)
            else:
                ops.append(eye)
        c_j = qt.tensor(ops)
        c.append(c_j)

    for op in c:
        c_dag.append(op.dag())

    for j in range(N-1):
        jp1 = (j + 1) % N
        H += c_dag[j] * c[jp1] + c_dag[jp1] * c[j]
    H = -t * H

    # Anfangszustand (Grundzustand)
    rho0 = qt.fock_dm([2]*N, [0]*N)


#######################
###
###  Hier fehlt jetzt der Lindblad-Term Teil (in c_ops)!
###
########################


    # Simulation:
    #mesolve(H, rho0, tlist, c_ops=[], e_ops=[], args={}, options={None,progress_bar:None}, _safe_mode=True)
    result = qt.mesolve(H, rho0, tlist, [], e_ops=[(c_dag[0] * c[0]),(c_dag[-1] * c[-1])], args={}) #,options={"progress_bar":True})
    ergebnisse = "Ergebnisse: \n"
    message1 = str([np.array2string(arr, precision=2, suppress_small=True) for arr in result.expect]) + "\n"
    message2 = str([np.array2string(arr, precision=2, suppress_small=True) for arr in result.states]) + "\n"
    write_to_output(ergebnisse + message1 + message2)












