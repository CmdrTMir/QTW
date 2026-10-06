from manim import *
import numpy as np
from numpy.linalg import eigh

class NanoribbonDispersion2(Scene):
    def construct(self):
        # --- 1. Parameter ---
        t = 1.0
        Ny = 20  # Breite des Nanoribbons
        dim = 2 * Ny

        M  = np.array([[0, 0], [0, 4*t]])
        Tx = np.array([[t, 1j*t], [1j*t, -t]])
        Ty = np.array([[t, t], [-t, -t]])

        # --- 2. Koordinatensystem ---
        x_min, x_max = -PI, PI
        y_min, y_max = -4, 8
        padding = 0.4

        axes = Axes(
            x_range=[x_min - padding, x_max + padding, PI/2],
            y_range=[y_min - padding, y_max + padding, 2],
            x_length=10,
            y_length=6,
            axis_config={
                "include_tip": False,
                "include_numbers": False,
                "font_size": 20,
                "color": WHITE
            },
        )

        tip_size = 0.08
        x_tip = Triangle(fill_opacity=1, color=WHITE).scale(tip_size)
        x_tip.move_to(axes.c2p(x_max + padding, 0)).rotate(-PI/2)
        y_tip = Triangle(fill_opacity=1, color=WHITE).scale(tip_size)
        y_tip.move_to(axes.c2p(0, y_max + padding))
        axes.add(x_tip, y_tip)
        axes_labels = axes.get_axis_labels(x_label="k_x", y_label="E")

        x_ticks = VGroup()
        x_labels = VGroup()
        for val, label in [(-PI, r"-\pi"), (-PI/2, r"-\pi/2"), (0, ""), (PI/2, r"\pi/2"), (PI, r"\pi")]:
            tick = Line(axes.c2p(val, 0), axes.c2p(val, -0.15), color=WHITE)
            lbl = MathTex(label, font_size=28).next_to(tick, DOWN, buff=0.1)
            x_ticks.add(tick)
            x_labels.add(lbl)

        y_ticks = VGroup()
        y_labels = VGroup()
        for val in [-4, 4, 8]:
            tick = Line(axes.c2p(0, val), axes.c2p(-0.08, val), color=WHITE)
            lbl = MathTex(str(val), font_size=20).next_to(tick, LEFT, buff=0.1)
            y_ticks.add(tick)
            y_labels.add(lbl)

        self.add(x_ticks, x_labels, y_ticks, y_labels)

        # --- 3. Berechnung der Bänder ---
        kx_vals = np.linspace(-PI, PI, 100)
        bulk_dots = VGroup()
        edge_dots = VGroup()

        for kx in kx_vals:
            # 1D-Hamiltonian für festes kx
            H_kx = np.zeros((dim, dim), dtype=complex)
            for m in range(Ny):
                # On-site + kx-Hopping
                H_kx[2*m:2*m+2, 2*m:2*m+2] = M - Tx * np.exp(1j*kx) - Tx.conj().T * np.exp(-1j*kx)
                # y-Hopping (Realraum)
                if m + 1 < Ny:
                    H_kx[2*(m+1):2*(m+1)+2, 2*m:2*m+2] = -Ty
                    H_kx[2*m:2*m+2, 2*(m+1):2*(m+1)+2] = -Ty.conj().T

            evals, evecs = eigh(H_kx)

            for i in range(dim):
                energy = evals[i]
                vec = evecs[:, i]
                # Wahrscheinlichkeit an den Rändern (m=0 und m=Ny-1)
                prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
                pt = axes.c2p(kx, energy)
                if prob_edges > 0.3:
                    dot = Dot(pt, radius=0.05, color=PINK)
                    edge_dots.add(dot)
                else:
                    dot = Dot(pt, radius=0.03, color=GREY_B, fill_opacity=0.5)
                    bulk_dots.add(dot)

        # --- 4. Animation ---
        title = Text("Nanoribbon Dispersion: \n Bulk vs. Edge States", font_size=30).to_corner(UL)
        legend = VGroup(
            Dot(color=GREY_B, radius=0.1), Text(" Bulk States", font_size=24),
            Dot(color=PINK, radius=0.1), Text(" Edge States", font_size=24)
        ).arrange(RIGHT, buff=0.3).to_corner(DR)
        self.play(FadeIn(legend))

        self.play(Write(title), Create(axes), Write(axes_labels), run_time=1.5)
        self.wait(0.5)

        # Bulk und Edge States einblenden
        self.play(FadeIn(bulk_dots), run_time=2)
        self.play(FadeIn(edge_dots), run_time=1.5)
        self.wait(3)
