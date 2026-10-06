from manim import *
import numpy as np
from numpy.linalg import eigh

# --- 0. Video- und Canvas-Größe anpassen ---
# Dies vergrößert den Koordinatenraum, sodass das Gitter kleiner relativ zum Bild wirkt.
# Für die tatsächliche Pixel-Auflösung nutzen Sie beim Rendern bitte -qh (720p) oder -qk (4k).
config.frame_width = 25
config.frame_height = 14.0625

class TimeEvo2(Scene):
    def construct(self):
        # --- 1. Hamiltonian ---
        t = 1.0
        N = 15
        dim = 2 * N * N

        def site_idx(m, n):
            return 2 * (m * N + n)

        H = np.zeros((dim, dim), dtype=complex)
        M  = np.array([[0, 0], [0, 4*t]])
        Tx = np.array([[t, 1j*t], [1j*t, -t]])
        Ty = np.array([[t, t], [-t, -t]])

        for m in range(N):
            for n in range(N):
                i = site_idx(m, n)
                H[i:i+2, i:i+2] += M
                if m + 1 < N:
                    j = site_idx(m + 1, n)
                    H[j:j+2, i:i+2] -= Tx
                    H[i:i+2, j:j+2] -= Tx.conj().T
                if n + 1 < N:
                    k = site_idx(m, n + 1)
                    H[k:k+2, i:i+2] -= Ty
                    H[i:i+2, k:k+2] -= Ty.conj().T

        # 1a. Defekt Masken vorbereiten
        H_defect_corner = H.copy()
        defect_sites_corner = set()
        defect_mask = np.ones(dim, dtype=complex)
        for m in range(3):
            for n in range(3):
                defect_sites_corner.add((m,n))
                i = site_idx(m, n)
                defect_mask[i:i+2] = 0
        mask_matrix = np.diag(defect_mask)
        H_defect_corner = mask_matrix @ H_defect_corner @ mask_matrix

        H_defect_edge = H.copy()
        defect_sites_edge = set()
        defect_mask = np.ones(dim, dtype=complex)
        for m in range(3):
            for n in range(5, 10):
                defect_sites_edge.add((m,n))
                i = site_idx(m, n)
                defect_mask[i:i+2] = 0
        mask_matrix = np.diag(defect_mask)
        H_defect_edge = mask_matrix @ H_defect_edge @ mask_matrix

        # Re-use Visualisation --- time evolution
        def show_evolution(H, title_text, defect_sites=None):
            if defect_sites is None:
                defect_sites = set()

            eigvalues, eigvectors = eigh(H)
            edge_masking = (eigvalues > 0) & (eigvalues < 4 * t)
            edge_indices = np.where(edge_masking)[0]

            hbar = 1
            E0 = 2 * t
            sigma_E = 0.3 * t
            T_max = 50.0
            N_frames = 150
            time_steps = np.linspace(0, T_max, N_frames)

            Psis = []
            coefficients_alpha = np.zeros(len(edge_indices), dtype=complex)
            for i, alpha in enumerate(edge_indices):
                weight = np.exp(-(eigvalues[alpha] - E0)**2 / (2 * sigma_E**2))
                coefficients_alpha[i] = weight

            for tau in time_steps:
                Psi_tau = np.zeros(dim, dtype=complex)
                for i, alpha in enumerate(edge_indices):
                    phase_time = np.exp(-1j * eigvalues[alpha] * tau / hbar)
                    Psi_tau += coefficients_alpha[i] * phase_time * eigvectors[:, alpha]
                norm = np.linalg.norm(Psi_tau)
                Psi_tau_norm = Psi_tau / norm if norm > 0 else Psi_tau
                Psis.append(Psi_tau_norm)

            # --- 2. Visuelles 15x15 Gitter ---
            spacing = 0.85
            circles = VGroup()
            arrows = VGroup()

            for m in range(N):
                for n in range(N):
                    circle = Circle(radius=0.4, color="#000080", fill_opacity=0.0, stroke_width=2.0, stroke_color=WHITE)
                    is_defect = (m, n) in defect_sites
                    if is_defect:
                        circle = Circle(radius=0.4, color="#d50514", fill_opacity=1, stroke_width=2.0, stroke_color=WHITE)
                    circle.move_to(n * RIGHT * spacing + m * UP * spacing)
                    circles.add(circle)

                    arrow = Line(LEFT * 0.3, RIGHT * 0.25).add_tip(tip_length=0.3, tip_width=0.2)
                    if is_defect:
                        arrow.set_color("#d50514")
                    else:
                        arrow.set_color(WHITE)
                    arrow.move_to(n * RIGHT * spacing + m * UP * spacing)
                    arrows.add(arrow)

            circles.move_to(ORIGIN)
            arrows.move_to(ORIGIN)

            title = Text("Time Evolution: ", font_size=40).to_corner(UL)
            subtitle = Text(title_text, font_size=36).next_to(title, DOWN, aligned_edge=LEFT, buff=0.2)
            frame_tracker = ValueTracker(0)
            prob_label = Text("Opacity = Probability \n Arrow = Phase", font_size=30).to_corner(DL)
            time_label = Text("t = ", font_size=40).to_corner(UR)
            time_label.shift(LEFT * 1.5)
            time_val = DecimalNumber(0, num_decimal_places=1, font_size=55).next_to(time_label, RIGHT, buff=0.3)
            time_val.add_updater(lambda d: d.set_value(frame_tracker.get_value() * (T_max / N_frames)))
            time_group = VGroup(time_label, time_val)

            self.play(Write(title), FadeIn(subtitle), FadeIn(circles), FadeIn(arrows), FadeIn(prob_label), FadeIn(time_group), run_time=1.5)
            self.wait(0.5)

            # --- 3. Der Uhrwerk-Updater ---
            def update_lattice(mob):
                idx = int(frame_tracker.get_value())
                idx = max(0, min(idx, N_frames - 1))

                psi_complex = Psis[idx].reshape(N, N, 2)
                for m in range(N):
                    for n in range(N):
                        if (m, n) in defect_sites:
                            continue

                        psi_s = psi_complex[m, n, 0]
                        psi_p = psi_complex[m, n, 1]

                        prob = np.abs(psi_s)**2 + np.abs(psi_p)**2
                        phase = np.angle(psi_s + psi_p)

                        circle = circles[m * N + n]
                        arrow = arrows[m * N + n]
                        opacity = min(prob * 15.0, 1.0)
                        circle.set_fill(opacity=opacity)
                        if opacity > 0.05:
                            arrow.set_opacity(1.0)
                            current_angle = arrow.get_angle()
                            diff = phase - current_angle
                            diff = (diff + np.pi) % (2 * np.pi) - np.pi
                            arrow.rotate(diff, about_point=circle.get_center())
                        else:
                            arrow.set_opacity(0.0)

            circles.add_updater(update_lattice)

            # --- 4. Die Animation ---
            self.play(
                frame_tracker.animate.set_value(N_frames - 1),
                run_time=32,
                rate_func=linear
            )
            self.wait(0.5)
            self.play(FadeOut(circles), FadeOut(arrows), FadeOut(title), FadeOut(subtitle), FadeOut(prob_label), FadeOut(time_group))

        # --- 5. Die verschiedenen Aufrufe ---
        show_evolution(H, "No defect")
        self.wait(0.5)
        show_evolution(H_defect_corner, "With defect at corner", defect_sites_corner)
        self.wait(0.5)
        show_evolution(H_defect_edge, "With defect at edge", defect_sites_edge)





