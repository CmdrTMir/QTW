from manim import *
import numpy as np

class BandHybridization(Scene):
    def construct(self):
#def BandHybridization(scene):
        # --- 1. Parameter-Setup ---
        a = 1.0
        e_s = 0.0
        e_p = 0.5
        # ValueTracker für die kontinuierliche Änderung
        t_ss_tracker = ValueTracker(0.0)
        t_pp_tracker = ValueTracker(0.0)
        t_sp_tracker = ValueTracker(0.0)
        t_ss_max = 1.0
        t_pp_max = -0.5
        t_sp_max = 0.6
        # --- 2. Das "Ein-Quadrant"-Koordinatensystem ---
        x_min, x_max = -PI, PI
        y_min, y_max = -2.5, 2.5
        # Ein unsichtbares Axes-Objekt nur für die Koordinatenumrechnung
        axes = Axes(
            x_range=[x_min, x_max],
            y_range=[y_min, y_max],
            x_length=10,
            y_length=6,
        )
        # Die eigentlichen Achsen (unten und links)
        x_axis_line = Line(axes.c2p(x_min, y_min), axes.c2p(x_max, y_min), stroke_width=2)
        y_axis_line = Line(axes.c2p(x_min, y_min), axes.c2p(x_min, y_max), stroke_width=2)
        # Kleine Pfeilspitzen am Ende der Achsen
        x_tip = Triangle(fill_opacity=1, color=WHITE).scale(0.08)
        x_tip.move_to(axes.c2p(x_max, y_min)).rotate(-PI/2)
        y_tip = Triangle(fill_opacity=1, color=WHITE).scale(0.08)
        y_tip.move_to(axes.c2p(x_min, y_max))
        # Achsen-Beschriftungen
        x_label = MathTex(r"k \text{ (wave number)}", font_size=30).next_to(x_axis_line, DOWN, buff=0.6)
        y_label = MathTex("E(k)", font_size=30).next_to(y_axis_line, LEFT, buff=1).rotate(PI/2)
        # X-Ticks und Labels (unten)
        x_ticks = VGroup()
        x_tick_labels = VGroup()
        for val, label in [(-PI, r"-\pi"), (-PI/2, r"-\pi / 2"), (0, "0"), (PI/2, r"\pi / 2"), (PI, r"\pi")]:
            tick = Line(axes.c2p(val, y_min), axes.c2p(val, y_min - 0.05))
            lbl = MathTex(label, font_size=30).next_to(tick, DOWN, buff=0.2)
            x_ticks.add(tick)
            x_tick_labels.add(lbl)
        # Y-Ticks und Labels (links)
        y_ticks = VGroup()
        y_tick_labels = VGroup()
        for val in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            tick = Line(axes.c2p(x_min, val), axes.c2p(x_min - 0.05, val))
            lbl = MathTex(f"{val:.1f}", font_size=30).next_to(tick, LEFT, buff=0.5)
            y_ticks.add(tick)
            y_tick_labels.add(lbl)

        def make_param_group(tracker, label_str):
            val = DecimalNumber(0.00, num_decimal_places=2, font_size=38)
            val.add_updater(lambda m: m.set_value(tracker.get_value()))
            return VGroup(MathTex(label_str, font_size=38), val).arrange(RIGHT, buff=0.1)
        t_ss_g = make_param_group(t_ss_tracker, r"t_{\text{ss}} = ")
        t_pp_g = make_param_group(t_pp_tracker, r"t_{\text{pp}} = ")
        t_sp_g = make_param_group(t_sp_tracker, r"t_{\text{sp}} = ")
        # Stapeln und positionieren
        params = VGroup(t_ss_g, t_pp_g, t_sp_g).arrange(LEFT, buff=0.5)
        params.move_to(axes.c2p(0, 2))
        # Alles auf einmal einblenden
        axis_group = VGroup(
            x_axis_line, y_axis_line, x_tip, y_tip,
            x_label, y_label,
            x_ticks, x_tick_labels,
            y_ticks, y_tick_labels,
            params
        )
        self.play(Create(axis_group), run_time=0.5)
        self.wait(0.5)
        # --- 3. Hilfsfunktion zur Berechnung der Bänder ---
        def get_bands_vgroup():
            t_ss = t_ss_tracker.get_value()
            t_pp = t_pp_tracker.get_value()
            t_sp = t_sp_tracker.get_value()
            t_ps = -t_sp
            k_vals = np.linspace(-PI, PI, 200)
            lines_lower = VGroup()
            lines_upper = VGroup()

            for i in range(len(k_vals) - 1):
                k1 = k_vals[i]
                k2 = k_vals[i+1]
                colors_lower = []
                colors_upper = []
                pts_lower = []
                pts_upper = []

                for k in [k1, k2]:
                    # Hamilton-Matrix aufbauen
                    H11 = e_s - 2 * t_ss * np.cos(k * a)
                    H22 = e_p - 2 * t_pp * np.cos(k * a)
                    H12 = -t_sp * np.exp(1j * k * a) - t_ps * np.exp(-1j * k * a)
                    H21 = np.conj(H12)
                    H = np.array([[H11, H12], [H21, H22]], dtype=complex)
                    evals, evecs = np.linalg.eigh(H)
                    E_minus, E_plus = evals[0], evals[1]
                    vec_minus, vec_plus = evecs[:, 0], evecs[:, 1]
                    # p-Anteil berechnen (2. Komponente des Eigenvektors)
                    weight_p_minus = np.abs(vec_minus[1])**2
                    weight_p_plus = np.abs(vec_plus[1])**2
                    color_minus = interpolate_color(TEAL, PINK, weight_p_minus)
                    color_plus = interpolate_color(TEAL, PINK, weight_p_plus)
                    pts_lower.append(axes.coords_to_point(k, E_minus))
                    pts_upper.append(axes.coords_to_point(k, E_plus))
                    colors_lower.append(color_minus)
                    colors_upper.append(color_plus)
                # Liniensegment erstellen mit der Durchschnittsfarbe der Endpunkte
                # (Manim kann keine Gradienten auf einzelnen Lines, also nehmen wir den Mittelwert)
                avg_color_lower = (np.array(colors_lower[0]) + np.array(colors_lower[1])) / 2
                avg_color_plus = (np.array(colors_upper[0]) + np.array(colors_upper[1])) / 2
                # Hex-Farbe für Manim Line zurückkonvertieren
                def rgb_to_hex(rgb):
                    return '#' + ''.join([hex(int(c*255))[2:].zfill(2) for c in rgb])

                line_lower = Line(pts_lower[0], pts_lower[1], color=rgb_to_hex(avg_color_lower), stroke_width=4)
                line_upper = Line(pts_upper[0], pts_upper[1], color=rgb_to_hex(avg_color_plus), stroke_width=4)
                lines_lower.add(line_lower)
                lines_upper.add(line_upper)

            return VGroup(lines_lower, lines_upper)

        # --- 4. Initialisierung und Updater ---
        bands = get_bands_vgroup()
        bands.add_updater(lambda m: m.become(get_bands_vgroup()))
        self.add(bands)
        legend = VGroup(
            Dot(color=TEAL, radius=0.1),
            Text(" s-orbital", font_size=28),
            Dot(color=PINK, radius=0.1),
            Text(" p-orbital", font_size=28),
            Dot(color=PURPLE_A, radius=0.1),
            Text(" hybridization", font_size=28)
        ).arrange(RIGHT, buff=0.3).to_corner(UL)
        self.play(FadeIn(legend))
        self.wait(0.2)
        # --- 5. Phase 1: t_ss und t_pp aufbauen (t_sp bleibt 0) ---
        self.play(
            t_ss_tracker.animate.set_value(t_ss_max),
            t_pp_tracker.animate.set_value(t_pp_max),
            run_time=5,
            rate_func=smooth
        )
        self.wait(0.5)
        # --- 6. Phase 2: t_sp erhöhen (Bandlücke öffnet sich) ---
        self.play(
            t_sp_tracker.animate.set_value(t_sp_max),
            run_time=5,
            rate_func=smooth
        )
        self.wait(1)
