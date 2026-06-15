import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from SimNatrium import Natrium
from SimLindblad import lindblad
from SimLindbladT import T_lindblad
from SimQutipLMEq import qutip_lindblad
from SimAnalytical import analytical
from SimSingleExcitation import single_excitation


class TabWithMode(ttk.Frame):
    def __init__(self, parent, tab_name, script_func, mode="plot", param_list=None):
        super().__init__(parent)
        self.script_func = script_func
        self.mode = mode
        self.params = param_list

        # Beschreibungsbox oben
        self.desc = tk.Text(self, height=4, wrap=tk.WORD)
        self.desc.pack(fill=tk.X, padx=5, pady=5)
        self.desc.insert(tk.END, f"Tab: {tab_name} (Modus: {mode})")

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
            if "gamma" in param_list: # gamma Parameter
                tk.Label(param_frame, text="γ: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.gamma_var = tk.DoubleVar(value=1.0)
                self.gamma_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.gamma_var, font=("Arial", 12), width=8)
                self.gamma_spin.pack(side=tk.LEFT, padx=5)
            if "kappa" in param_list: # kappa Parameter
                tk.Label(param_frame, text="κ: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.kappa_var = tk.DoubleVar(value=1.0)
                self.kappa_spin = tk.Spinbox(param_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.kappa_var, font=("Arial", 12), width=8)
                self.kappa_spin.pack(side=tk.LEFT, padx=5)
            if "d" in param_list: # d Parameter
                tk.Label(param_frame, text="d: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
                self.d_var = tk.IntVar(value=1)
                self.d_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.d_var, font=("Arial", 12), width=5)
                self.d_spin.pack(side=tk.LEFT, padx=5)
            self.param_vars = {}
            if hasattr(self, 'N_var'): self.param_vars['N'] = self.N_var
            if hasattr(self, 't_var'): self.param_vars['t'] = self.t_var
            if hasattr(self, 'gamma_var'): self.param_vars['gamma'] = self.gamma_var
            if hasattr(self, 'kappa_var'): self.param_vars['kappa'] = self.kappa_var
            if hasattr(self, 'd_var'): self.param_vars['d'] = self.d_var

        # Buttons zum Ausführen
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        self.run_button = tk.Button(button_frame, text="Ausführen", font=("Arial", 11, "bold"), command=self.execute_script)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.steady_button = tk.Button(button_frame, text="Steady State", font=("Arial", 11, "bold"), command=self.execute_steady_state)
        self.steady_button.pack(side=tk.LEFT, padx=5)

        self.figure, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.show_empty_plot()
        # Initialen leeren Text-Bereich anzeigen
        self.output_text = tk.Text(self, height=15, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def show_empty_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Noch nichts ausgeführt\nKlicken Sie auf 'Ausführen'",
                     ha='center', va='center', fontsize=12, alpha=0.5)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()

    def write_to_output(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def execute_script(self):
        if hasattr(self, 'param_vars'):
            params = {name: var.get() for name, var in self.param_vars.items()}
            params["mode"] = "time"
            if self.mode == "plot":
                self.ax.clear()
                self.script_func(self.ax, params)
                self.canvas.draw()
            else:
                self.output_text.delete(1.0, tk.END)
                self.script_func(self.write_to_output, params)
        else:
            if self.mode == "plot":
                self.ax.clear()
                self.script_func(self.ax)
                self.canvas.draw()
            else:
                self.output_text.delete(1.0, tk.END)
                self.script_func(self.write_to_output)

    def execute_steady_state(self):
        if hasattr(self, 'param_vars'):
            params = {name: var.get() for name, var in self.param_vars.items()}
            params["mode"] = "steady"
            self.ax.clear()
            self.script_func(self.ax, params)
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
        tab1 = TabWithMode(sub_notebook1, "Lindblad", lindblad, "plot", ["N", "t", "gamma"])
        sub_notebook1.add(tab1, text="Lindblad")
        tab2 = TabWithMode(sub_notebook1, "variable T", T_lindblad, "plot", ["N", "t", "gamma", "d"])
        sub_notebook1.add(tab2, text="variable T")
        tab4 = TabWithMode(sub_notebook1, "Analytical", analytical, "plot", ["N", "t", "gamma"])
        sub_notebook1.add(tab4, text="Analytical")
        tab6 = TabWithMode(sub_notebook1, "Qutip LMEq", qutip_lindblad, "text", ["N", "t", "gamma"])
        sub_notebook1.add(tab6, text="Qutip LMEq")
        tab7 = TabWithMode(sub_notebook1, "SingleExcitation", single_excitation, "plot", ["N", "t", "gamma", "kappa"])
        sub_notebook1.add(tab7, text="SingleExcitation")

        # === Gruppe 2: 2-BandModels ===
        group2_frame = ttk.Frame(self.notebook)
        self.notebook.add(group2_frame, text="2-BandModels")
        sub_notebook2 = ttk.Notebook(group2_frame)
        sub_notebook2.pack(fill=tk.BOTH, expand=True)

        # Tabs in Gruppe 2
        tab3 = TabWithMode(sub_notebook2, "Natrium", Natrium, "plot", None)
        sub_notebook2.add(tab3, text="Natrium")

        # === Gruppe 3: Tests ===
        group3_frame = ttk.Frame(self.notebook)
        self.notebook.add(group3_frame, text="Tests")
        sub_notebook3 = ttk.Notebook(group3_frame)
        sub_notebook3.pack(fill=tk.BOTH, expand=True)

        tab5 = TabWithMode(sub_notebook3, "Sinus", script1, "plot", None)
        sub_notebook3.add(tab5, text="Sinus")



if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()
