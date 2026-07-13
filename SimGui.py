import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from SimNatrium import Natrium
from SimLindblad import lindblad
from SimLindbladT import T_lindblad
from SimQutipLMEq import qutip_lindblad
#from SimAnalytical import analytical
from SimSingleExcitation import single_excitation
from SimDimReduction import dim_reduction
from SimKette2DPlot import chain_to_2D
from Sim2D import dimRed_2D
from Sim2DAni import ani_2D
from SimEigenvals2D import eigenvals_2D
from Sim1D2band import twoband_1D


class TabWithMode(ttk.Frame):
    def __init__(self, parent, tab_name, script_func, param_list=None, n_axes=1):
        super().__init__(parent)
        self.script_func = script_func
        self.params = param_list
        self.n_axes = n_axes

        # Beschreibungsbox oben
        self.desc = tk.Text(self, height=4, wrap=tk.WORD)
        self.desc.pack(fill=tk.X, padx=5, pady=5)
        self.desc.insert(tk.END, f"Tab: {tab_name}")

        if param_list:
            # Parameter-Frame
            param_frame = tk.Frame(self)#, bg="#FAF0E6") # Leder   bg="#FDF5E6" -- Altweiß
            param_frame.pack(fill=tk.X, padx=5, pady=5)
            if "N" in param_list: # N Parameter
                tk.Label(param_frame, text="N: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.N_var = tk.IntVar(value=2)
                self.N_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.N_var, font=("Arial", 12), width=5)
                self.N_spin.pack(side=tk.LEFT, padx=5)
            if "t" in param_list: # t Parameter
                tk.Label(param_frame, text="t: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.t_var = tk.DoubleVar(value=2.0)
                self.t_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.t_var, font=("Arial", 12), width=8)
                self.t_spin.pack(side=tk.LEFT, padx=5)
            if "t_v" in param_list: # vertical t Parameter
                tk.Label(param_frame, text="t_v: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.t_v_var = tk.DoubleVar(value=2.0)
                self.t_v_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.t_v_var, font=("Arial", 12), width=8)
                self.t_v_spin.pack(side=tk.LEFT, padx=5)
            if "gamma" in param_list: # gamma Parameter
                tk.Label(param_frame, text="γ (out): ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.gamma_var = tk.DoubleVar(value=1.0)
                self.gamma_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.gamma_var, font=("Arial", 12), width=8)
                self.gamma_spin.pack(side=tk.LEFT, padx=5)
            if "kappa" in param_list: # kappa Parameter
                tk.Label(param_frame, text="κ (in): ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.kappa_var = tk.DoubleVar(value=1.0)
                self.kappa_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.kappa_var, font=("Arial", 12), width=8)
                self.kappa_spin.pack(side=tk.LEFT, padx=5)
            if "d" in param_list: # d Parameter
                tk.Label(param_frame, text="d: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.d_var = tk.IntVar(value=1)
                self.d_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.d_var, font=("Arial", 12), width=5)
                self.d_spin.pack(side=tk.LEFT, padx=5)
            if "tf" in param_list: # tf Parameter
                tk.Label(param_frame, text="tf: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.tf_var = tk.IntVar(value=100)
                self.tf_spin = tk.Spinbox(param_frame, from_=50, to=1000, textvariable=self.tf_var, font=("Arial", 12), width=5)
                self.tf_spin.pack(side=tk.LEFT, padx=5)
            # Params for 2band model:
            if "t_ss" in param_list:
                tk.Label(param_frame, text="t_ss: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.tss_var = tk.DoubleVar(value=1.0)
                self.tss_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.tss_var, font=("Arial", 12), width=8)
                self.tss_spin.pack(side=tk.LEFT, padx=5)
            if "t_pp" in param_list:
                tk.Label(param_frame, text="t_pp: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.tpp_var = tk.DoubleVar(value=0.5)
                self.tpp_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.tpp_var, font=("Arial", 12), width=8)
                self.tpp_spin.pack(side=tk.LEFT, padx=5)
            if "t_sp" in param_list:
                tk.Label(param_frame, text="t_sp: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.tsp_var = tk.DoubleVar(value=0.0)
                self.tsp_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.tsp_var, font=("Arial", 12), width=8)
                self.tsp_spin.pack(side=tk.LEFT, padx=5)
            if "e_s" in param_list:
                tk.Label(param_frame, text="ε_s: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.es_var = tk.DoubleVar(value=0.0)
                self.es_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.es_var, font=("Arial", 12), width=8)
                self.es_spin.pack(side=tk.LEFT, padx=5)
            if "e_p" in param_list:
                tk.Label(param_frame, text="ε_p: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.ep_var = tk.DoubleVar(value=2.0)
                self.ep_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.ep_var, font=("Arial", 12), width=8)
                self.ep_spin.pack(side=tk.LEFT, padx=5)
            self.param_vars = {}
            if hasattr(self, 'N_var'): self.param_vars['N'] = self.N_var
            if hasattr(self, 't_var'): self.param_vars['t'] = self.t_var
            if hasattr(self, 't_v_var'): self.param_vars['t_v'] = self.t_v_var
            if hasattr(self, 'gamma_var'): self.param_vars['gamma'] = self.gamma_var
            if hasattr(self, 'kappa_var'): self.param_vars['kappa'] = self.kappa_var
            if hasattr(self, 'd_var'): self.param_vars['d'] = self.d_var
            if hasattr(self, 'tf_var'): self.param_vars['tf'] = self.tf_var
            if hasattr(self, 'tss_var'): self.param_vars['t_ss'] = self.tss_var
            if hasattr(self, 'tpp_var'): self.param_vars['t_pp'] = self.tpp_var
            if hasattr(self, 'tsp_var'): self.param_vars['t_sp'] = self.tsp_var
            if hasattr(self, 'es_var'): self.param_vars['e_s'] = self.es_var
            if hasattr(self, 'ep_var'): self.param_vars['e_p'] = self.ep_var

        # Buttons zum Ausführen
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)
        self.run_button = tk.Button(button_frame, text="Ausführen", font=("Arial", 11, "bold"), command=self.execute_script)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.steady_button = tk.Button(button_frame, text="Steady State", font=("Arial", 11, "bold"), command=self.execute_steady_state)
        self.steady_button.pack(side=tk.LEFT, padx=5)

        if n_axes == 2:
            self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(6, 4))
            self.ax = self.ax1
        else:
            self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.show_empty_plot()
        # Initialen leeren Text-Bereich anzeigen
        self.output_text = tk.Text(self, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def show_empty_plot(self):
        self.ax.clear()
        if self.n_axes == 2:
            self.ax2.clear()
            self.ax.text(0.5, 0.5, "Noch nichts ausgeführt\nKlicken Sie auf 'Ausführen'",
                        ha='center', va='center', fontsize=12, alpha=0.5)
            self.ax.set_xlim(0,1); self.ax.set_ylim(0,1); self.ax.axis('off')
            self.ax2.set_xlim(0,1); self.ax2.set_ylim(0,1); self.ax2.axis('off')

        self.ax.text(0.5, 0.5, "Noch nichts ausgeführt\nKlicken Sie auf 'Ausführen'",
                        ha='center', va='center', fontsize=12, alpha=0.5)
        self.ax.set_xlim(0,1); self.ax.set_ylim(0,1); self.ax.axis('off')
        self.canvas.draw()

    def write_to_output(self, message, color="black"):
        tag_name = f"color_{color}"
        self.output_text.tag_config(tag_name, foreground=color)
        self.output_text.insert(tk.END, message + "\n", tag_name)
        self.output_text.see(tk.END)

    def execute_script(self):
        if self.n_axes == 2:
            self.ax2.clear()
        self.ax.clear()
        self.output_text.delete(1.0, tk.END)
        if hasattr(self, 'param_vars'):
            params = {name: var.get() for name, var in self.param_vars.items()}
            params["mode"] = "time"
            if self.n_axes == 2:
                self.script_func(self.ax, self.ax2, self.write_to_output, params)
            else:
                self.script_func(self.ax, self.write_to_output, params)
        else:
            self.script_func(self.ax, self.write_to_output)
        self.canvas.draw()

    def execute_steady_state(self):
        if hasattr(self, 'param_vars'):
            params = {name: var.get() for name, var in self.param_vars.items()}
            params["mode"] = "steady"
            self.ax.clear()
            self.output_text.delete(1.0, tk.END)
            self.script_func(self.ax, self.write_to_output, params)
            self.canvas.draw()

# ------------------------------------------------------------
def script1(ax):
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label="Skript 1: Sinus")
    ax.legend()

# ------------------------------------------------------------
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI for multiple tasks")
        self.root.configure(bg="#000080") #navy

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === Gruppe 1: Full Dims ===
        group1_frame = ttk.Frame(self.notebook)
        self.notebook.add(group1_frame, text="Full Dims")
        sub_notebook1 = ttk.Notebook(group1_frame)
        sub_notebook1.pack(fill=tk.BOTH, expand=True)

        # Tabs in Gruppe 1
        tab1 = TabWithMode(sub_notebook1, "Lindblad", lindblad, ["N", "t", "gamma"])
        sub_notebook1.add(tab1, text="Lindblad")
        tab2 = TabWithMode(sub_notebook1, "variable T", T_lindblad, ["N", "t", "gamma", "d"])
        sub_notebook1.add(tab2, text="variable T")
        #tab4 = TabWithMode(sub_notebook1, "Analytical", analytical, ["N", "t", "gamma"])
        #sub_notebook1.add(tab4, text="Analytical")
        tab6 = TabWithMode(sub_notebook1, "Qutip LMEq", qutip_lindblad, ["N", "t", "gamma"])
        sub_notebook1.add(tab6, text="Qutip LMEq")
        tab7 = TabWithMode(sub_notebook1, "SingleExcitation", single_excitation, ["N", "t", "gamma", "kappa", "tf"])
        sub_notebook1.add(tab7, text="SingleExcitation")

        # === Gruppe 2: 2-BandModels ===
        group2_frame = ttk.Frame(self.notebook)
        self.notebook.add(group2_frame, text="2-BandModels")
        sub_notebook2 = ttk.Notebook(group2_frame)
        sub_notebook2.pack(fill=tk.BOTH, expand=True)

        # Tabs in Gruppe 2
        tab3 = TabWithMode(sub_notebook2, "Natrium", Natrium, None)
        sub_notebook2.add(tab3, text="Natrium")
        tab3a = TabWithMode(sub_notebook2, "1D2band", twoband_1D, ["t_ss", "t_pp", "t_sp", "e_s", "e_p"])
        sub_notebook2.add(tab3a, text="1D2band")

        # === Gruppe 3: Tests ===
        group3_frame = ttk.Frame(self.notebook)
        self.notebook.add(group3_frame, text="Tests")
        sub_notebook3 = ttk.Notebook(group3_frame)
        sub_notebook3.pack(fill=tk.BOTH, expand=True)

        # Tabs in Gruppe 3
        tab5 = TabWithMode(sub_notebook3, "Sinus", script1, None)
        sub_notebook3.add(tab5, text="Sinus")


        # === Gruppe 4: (N+1)x(N+1) ===
        group4_frame = ttk.Frame(self.notebook)
        self.notebook.add(group4_frame, text="(N+1)x(N+1)")
        sub_notebook4 = ttk.Notebook(group4_frame)
        sub_notebook4.pack(fill=tk.BOTH, expand=True)

        #Tabs in Gruppe 4
        tab8 = TabWithMode(sub_notebook4, "DimReduction", dim_reduction, ["N"])
        sub_notebook4.add(tab8, text="DimReduction")
        tab9 = TabWithMode(sub_notebook4, "Kette zu 2D", chain_to_2D, ["N"])
        sub_notebook4.add(tab9, text="Kette zu 2D")
        tab10 = TabWithMode(sub_notebook4, "2D", dimRed_2D, ["N", "t", "t_v", "tf"])
        sub_notebook4.add(tab10, text="2D")
        tab12 = TabWithMode(sub_notebook4, "eigenvals_2D", eigenvals_2D, ["N", "t", "t_v"], 2)
        sub_notebook4.add(tab12, text="eigenvals_2D")
        #tab11 = TabWithMode(sub_notebook4, "2D Animation", ani_2D, ["N", "t", "gamma", "kappa", "tf"])
        #sub_notebook4.add(tab11, text="2D Animation")




if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()
