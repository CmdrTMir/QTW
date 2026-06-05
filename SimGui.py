import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from SimNatrium import Natrium
from SimLindblad import lindblad
from SimQutipLMEq import qutip_lindblad
from SimAnalytical import analytical


class TabWithMode(ttk.Frame):
    def __init__(self, parent, tab_name, script_func, mode="plot", has_params=False):
        super().__init__(parent)
        self.script_func = script_func
        self.mode = mode
        self.params = has_params

        # Beschreibungsbox oben
        self.desc = tk.Text(self, height=4, wrap=tk.WORD)
        self.desc.pack(fill=tk.X, padx=5, pady=5)
        self.desc.insert(tk.END, f"Tab: {tab_name} (Modus: {mode})")

        if has_params:
            # Parameter-Frame
            param_frame = tk.Frame(self)#, bg="#FAF0E6") # Leder   bg="#FDF5E6" -- Altweiß
            param_frame.pack(fill=tk.X, padx=5, pady=5)
            # N Parameter
            tk.Label(param_frame, text="N: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
            self.N_var = tk.IntVar(value=2)
            self.N_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.N_var, font=("Arial", 12), width=5)
            self.N_spin.pack(side=tk.LEFT, padx=5)
            # t Parameter
            tk.Label(param_frame, text="t: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
            self.t_var = tk.IntVar(value=2)
            self.t_spin = tk.Spinbox(param_frame, from_=1, to=100, textvariable=self.t_var, font=("Arial", 12), width=5)
            self.t_spin.pack(side=tk.LEFT, padx=5)
            # gamma Parameter
            tk.Label(param_frame, text="γ: ", font=("Arial", 12)).pack(side=tk.LEFT, padx=2)
            self.gamma_var = tk.IntVar(value=1)
            self.gamma_spin = tk.Spinbox(param_frame, from_=0, to=100, textvariable=self.gamma_var, font=("Arial", 12), width=5)
            self.gamma_spin.pack(side=tk.LEFT, padx=5)

        # Button zum Ausführen
        self.run_button = tk.Button(self, text="Ausführen", font=("Arial", 11, "bold"), command=self.execute_script)
        self.run_button.pack(pady=5)

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
        if self.params:
            N = self.N_var.get()
            t = self.t_var.get()
            gamma = self.gamma_var.get()
            if self.mode == "plot":
                self.ax.clear()
                self.script_func(self.ax, N, t, gamma)
                self.canvas.draw()
            else:
                self.output_text.delete(1.0, tk.END)
                self.script_func(self.write_to_output, N, t, gamma)
        else:
            if self.mode == "plot":
                self.ax.clear()
                self.script_func(self.ax)
                self.canvas.draw()
            else:
                self.output_text.delete(1.0, tk.END)
                self.script_func(self.write_to_output)

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

        self.tabs = {}
        tab_configs = [
            ("Sinus", script1, "plot", False),
            ("Natrium", Natrium, "plot", False),
            ("Lindblad", lindblad, "plot", True),
            ("Qutip LMEq", qutip_lindblad, "text", True),
            ("Analytical", analytical, "plot", True)
        ]

        for name, func, mode, has_params in tab_configs:
            tab = TabWithMode(self.notebook, name, func, mode, has_params)
            self.notebook.add(tab, text=name)
            self.tabs[name] = tab

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()
