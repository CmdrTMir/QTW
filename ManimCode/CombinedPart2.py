from manim import *
import numpy as np
from Band import BandHybridization

class Combined(Scene):
    def hop_particle(self, particle, start, end, positions=None):
        # Die Schrödinger-Hopping-Funktion: 1D ohne positions, 2D mit positions.
        arrow_angle = -2
        for x in range(start, end):
            if positions is not None:
                arrow_angle = -1.5
                start_pt = positions[x]
                end_pt = positions[x+1]
                # Berechne den senkrechten Vektor für den 2D-Bogen
                diff = end_pt - start_pt
                perp = np.array([-diff[1], diff[0], 0])
                norm = np.linalg.norm(perp)
                if norm > 0:
                    perp = perp / norm * 0.1
                else:
                    perp = UP * 0.1
            else:
                # Exakter 1D-Fallback aus Ihrem Originalcode
                start_pt = RIGHT * x
                end_pt = RIGHT * (x + 1)
                perp = UP * 0.1

            arrow = ArcBetweenPoints(start_pt, end_pt + perp, angle=arrow_angle, color=TEAL)
            arrow.add_tip(tip_length=0.15)
            t = Text("t", font_size=24).next_to(arrow, perp * 1.5)
            arrow_group = VGroup(arrow, t)

            self.play(Create(arrow_group), run_time=0.4)
            self.play(FadeOut(particle), run_time=0.3)
            particle.move_to(end_pt)
            self.play(FadeIn(particle), run_time=0.3)
            self.play(FadeOut(arrow_group), run_time=0.4)

    def construct(self):
        # A: --- 1D setup ---
        line = Line(LEFT * 3.4, RIGHT * 3.4, color=GRAY)
        self.add(line)
        # Sites 1-5
        sites = VGroup()
        for x in range(-2, 3):
            site = Dot(RIGHT * x, radius=0.1)
            sites.add(site)
        self.add(sites)
        # Rand-Sites
        left_0 = Dot(LEFT * 3, radius=0.1, fill_opacity=0.35)
        right_0 = Dot(RIGHT * 3, radius=0.1, fill_opacity=0.35)
        left_0_label = Text("0", font_size=24).move_to(RIGHT * -3 + DOWN * 0.4)
        right_0_label = Text("0", font_size=24).move_to(RIGHT * 3 + DOWN * 0.4)
        self.add(left_0, right_0, left_0_label, right_0_label)
        # Site-Beschriftungen
        labels = [1, 2, 3, 4, 5]
        labels_group = VGroup()
        for x, label in zip(range(-2, 3), labels):
            labels_group.add(Text(str(label), font_size=24).next_to(RIGHT * x, DOWN * 1.1))
        self.add(labels_group)
        # Gitterabstand a
        a_lines = VGroup(Line(RIGHT * -2 + UP * 0.65, RIGHT * -2 + UP * 0.95, stroke_width=3),
                         Line(RIGHT * -1 + UP * 0.65, RIGHT * -1 + UP * 0.95, stroke_width=3),
                         Line(RIGHT * -2 + UP * 0.8, RIGHT * -1 + UP * 0.8, stroke_width=3))
        a_label = Text("a", font_size=24).move_to(RIGHT * -1.5 + UP * 0.95)
        self.add(a_lines, a_label)
        # j und j+1 auf Site 3 und 4
        circle3 = Circle(radius=0.22).move_to(RIGHT * 0 + DOWN * 0.4)
        circle4 = Circle(radius=0.22).move_to(RIGHT * 1 + DOWN * 0.4)
        j_labels = VGroup(circle3, circle4,
                         Text("j", font_size=20).next_to(circle3, RIGHT * 0.23 + DOWN * 0.04),
                         Text("j+1", font_size=20).next_to(circle4, RIGHT * 0.23 + DOWN * 0.04))
        self.add(j_labels)
        # Begrenzungslinien
        boundaries = VGroup(DashedLine(UP * 1.3 + LEFT * 2.6, DOWN + LEFT * 2.6, color=GRAY),
                            DashedLine(UP * 1.3 + RIGHT * 2.6, DOWN + RIGHT * 2.6, color=GRAY))
        self.add(boundaries)
        # Teilchen
        particle = Dot(RIGHT * 0, radius=0.12, color=RED_A)
        self.add(particle)
        self.hop_particle(particle, 0, 2)
        arrow = ArcBetweenPoints(RIGHT * 2, RIGHT * 3 + UP * 0.1, angle=-2, color=TEAL)
        arrow.add_tip(tip_length=0.15)
        t = Text("t", font_size=24).next_to(arrow, UP * 0.15)
        arrow_group = VGroup(arrow, t)
        self.play(Create(arrow_group), run_time=0.4)
        self.play(FadeOut(particle), run_time=0.3)
        particle.move_to(RIGHT * 3)
        particle_left = Dot(LEFT * 3, radius=0.12, color=RED_A)
        self.play(FadeIn(particle), FadeIn(particle_left), run_time=0.3)
        self.play(FadeOut(arrow_group), run_time=0.4)
        self.play(FadeOut(particle), run_time=0.4)
        # Linkes Teilchen weiter nach rechts: 0 -> 1 -> 2 -> 3
        particle = particle_left
        self.hop_particle(particle, -3, 0)

        # B: --- Open System ---
        self.play(FadeOut(particle), FadeOut(particle_left), FadeOut(left_0_label), FadeOut(right_0_label),
                  FadeOut(a_lines), FadeOut(a_label), FadeOut(j_labels), FadeOut(boundaries),
                  FadeOut(left_0), FadeOut(right_0), run_time=0.5)
        self.wait(0.5)
        explanation = Text("Open system", font_size=30).to_edge(UP)
        self.play(FadeIn(explanation), run_time=0.5)
        self.wait(1)
        # Pump
        pump_arrow = Arrow(LEFT * 3.5, LEFT * 2.1, color=BLUE, buff=0)
        pump_label = Text("Pump", font_size=24).next_to(pump_arrow, UP * 0.2)
        self.play(Create(pump_arrow), FadeIn(pump_label), run_time=0.5)
        particle = Dot(LEFT * 2, radius=0.12, color=RED_A)
        self.play(FadeIn(particle), run_time=0.3)
        self.play(FadeOut(pump_arrow), FadeOut(pump_label), run_time=0.3)
        self.hop_particle(particle, -2, 2)
        # Drain
        drain_arrow = Arrow(RIGHT * 2.1, RIGHT * 3.5, color=BLUE, buff=0)
        drain_label = Text("Drain", font_size=24).next_to(drain_arrow, UP * 0.2)
        self.play(Create(drain_arrow), FadeIn(drain_label), run_time=0.5)
        self.play(FadeOut(particle), run_time=0.5)
        self.play(FadeOut(drain_arrow), FadeOut(drain_label), run_time=0.3)
        self.wait(1)

        # C: --- 2D array (Schlangen-Muster) ---
        self.play(FadeOut(explanation), FadeOut(sites), FadeOut(labels_group), FadeOut(line), run_time=0.5)
        transition_text = Text("from 1D to a topological 2D mapping", font_size=30)
        self.play(FadeIn(transition_text), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(transition_text), run_time=0.5)
        # 2D set-up (Schlangen-Muster)
        positions_2d = [
            LEFT * 2 + UP * 2,   # 1: Top-Left
            UP * 2,              # 2: Top-Mid
            RIGHT * 2 + UP * 2,  # 3: Top-Right
            RIGHT * 2,           # 4: Mid-Right
            ORIGIN,              # 5: Mid-Mid
            LEFT * 2,            # 6: Mid-Left
            LEFT * 2 + DOWN * 2, # 7: Bot-Left
            DOWN * 2,            # 8: Bot-Mid
            RIGHT * 2 + DOWN * 2 # 9: Bot-Right
        ]
        sites_2d = VGroup()
        labels_2d = VGroup()
        for i, pos in enumerate(positions_2d):
            site = Dot(pos, radius=0.1)
            sites_2d.add(site)
            label = Text(str(i+1), font_size=24).next_to(pos, DOWN * 1.1)
            labels_2d.add(label)
        self.add(sites_2d, labels_2d)
        particle_2d = Dot(positions_2d[0], radius=0.12, color=RED_A)
        self.add(particle_2d)
        # Pump bei Site 1
        pump_arrow_2d = Arrow(positions_2d[0] + UP*1.2 + LEFT*1.2, positions_2d[0] + UP*0.2 + LEFT*0.2, color=BLUE, buff=0)
        pump_label_2d = Text("Pump", font_size=24).next_to(pump_arrow_2d, UP*0.2)
        self.play(Create(pump_arrow_2d), FadeIn(pump_label_2d), run_time=0.5)
        self.play(FadeOut(pump_arrow_2d), FadeOut(pump_label_2d), run_time=0.3)
        # Hopping durch das 2D-System (Index 0 bis 8, also Site 1 bis 9)
        self.hop_particle(particle_2d, 0, 8, positions=positions_2d)
        # Drain bei Site 9
        drain_arrow_2d = Arrow(positions_2d[8] + DOWN*0.2 + RIGHT*0.2, positions_2d[8] + DOWN*1.2 + RIGHT*1.2, color=BLUE, buff=0)
        drain_label_2d = Text("Drain", font_size=24).next_to(drain_arrow_2d, DOWN*0.2)
        self.play(Create(drain_arrow_2d), FadeIn(drain_label_2d), run_time=0.5)
        self.play(FadeOut(particle_2d), run_time=0.5)
        self.play(FadeOut(drain_arrow_2d), FadeOut(drain_label_2d), run_time=0.3)
        self.wait(1)

        # D: --- normal 2D array  ---
        self.play(FadeOut(sites_2d), FadeOut(labels_2d), FadeOut(particle_2d), run_time=0.5)
        transition_text_3 = Text("normal 2D mapping", font_size=30)
        self.play(FadeIn(transition_text_3), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(transition_text_3), run_time=0.5)
        positions_normal = [
            LEFT * 2 + UP * 2,   # 1: Top-Left
            UP * 2,              # 2: Top-Mid
            RIGHT * 2 + UP * 2,  # 3: Top-Right
            LEFT * 2,            # 4: Mid-Left
            ORIGIN,              # 5: Mid-Mid
            RIGHT * 2,           # 6: Mid-Right
            LEFT * 2 + DOWN * 2, # 7: Bot-Left
            DOWN * 2,            # 8: Bot-Mid
            RIGHT * 2 + DOWN * 2 # 9: Bot-Right
        ]
        sites_normal = VGroup()
        labels_normal = VGroup()
        for i, pos in enumerate(positions_normal):
            site = Dot(pos, radius=0.1)
            sites_normal.add(site)
            label = Text(str(i+1), font_size=24).next_to(pos, DOWN * 1.1)
            labels_normal.add(label)
        self.add(sites_normal, labels_normal)
        # Hilfsfunktion für sauberes Hopping
        def animate_path(particle, path_indices, label_text, color=TEAL):
            for i in range(len(path_indices) - 1):
                start_pt = positions_normal[path_indices[i]]
                end_pt = positions_normal[path_indices[i+1]]
                diff = end_pt - start_pt
                perp = np.array([-diff[1], diff[0], 0])
                norm = np.linalg.norm(perp)
                if norm > 0:
                    perp = perp / norm * 0.1
                else:
                    perp = UP * 0.1
                arrow = ArcBetweenPoints(start_pt, end_pt + perp, angle=-1.5, color=color)
                arrow.add_tip(tip_length=0.15)
                t_label =  MathTex(label_text, font_size=30).next_to(arrow, perp * 2, buff=0.3)
                arrow_group = VGroup(arrow, t_label)

                self.play(Create(arrow_group), run_time=0.4)
                self.play(FadeOut(particle), run_time=0.3)
                particle.move_to(end_pt)
                self.play(FadeIn(particle), run_time=0.3)
                self.play(FadeOut(arrow_group), run_time=0.4)
        # Pump bei Site 1
        pump_arrow_normal = Arrow(positions_normal[0] + UP*1.2 + LEFT*1.2,
                                  positions_normal[0] + UP*0.2 + LEFT*0.2,
                                  color=BLUE, buff=0)
        pump_label_normal = Text("Pump", font_size=24).next_to(pump_arrow_normal, UP*0.2)
        self.play(Create(pump_arrow_normal), FadeIn(pump_label_normal), run_time=0.5)
        particle_pump = Dot(positions_normal[0], radius=0.12, color=RED_A)
        self.play(FadeIn(particle_pump), run_time=0.3)
        self.play(FadeOut(pump_arrow_normal), FadeOut(pump_label_normal), FadeOut(particle_pump), run_time=0.3)
        # Hopping t_h (1 -> 2 -> 3)
        particle_h = Dot(positions_normal[0], radius=0.12, color=RED_A)
        self.add(particle_h)
        animate_path(particle_h, [0, 1, 2], r"\mathbf{t_h}")
        self.play(FadeOut(particle_h), run_time=0.4)
        # Hopping t_v (3 -> 6 -> 9)
        particle_v = particle_h
        self.add(particle_v)
        animate_path(particle_v, [2, 5, 8], r"\mathbf{t_v}", color=GREEN)
        self.play(FadeOut(particle_v), run_time=0.4)
        # Drain bei Site 9
        drain_arrow_normal = Arrow(positions_normal[8] + DOWN*0.2 + RIGHT*0.2,
                                   positions_normal[8] + DOWN*1.2 + RIGHT*1.2,
                                   color=BLUE, buff=0)
        drain_label_normal = Text("Drain", font_size=24).next_to(drain_arrow_normal, DOWN*0.2)
        self.play(Create(drain_arrow_normal), FadeIn(drain_label_normal), run_time=0.5)
        self.play(FadeOut(drain_arrow_normal), FadeOut(drain_label_normal), run_time=0.3)
        self.wait(1)

        # E: --- Tow orbital model ---
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.5)
        transition_text_orbital = Text("two orbital model", font_size=30)
        self.play(FadeIn(transition_text_orbital), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(transition_text_orbital), run_time=0.5)
        site_positions = [LEFT * 4, ORIGIN, RIGHT * 4]
        r_s = 0.25
        r_p = 0.3
        def create_site(position):
            site_group = VGroup()
            # p-Orbitale zuerst zeichnen, damit s-Orbital die Überlappung verdeckt
            p_top = Circle(radius=r_p, color=PINK).move_to(position + UP * r_p)
            p_bottom = Circle(radius=r_p, color=PINK).move_to(position + DOWN * r_p)
            # s-Orbital in der Mitte (Süd- und Nordpol der p-Kreise berühren sich hier)
            s_orbital = Circle(radius=r_s, color=TEAL, fill_color=TEAL, fill_opacity=0.5).move_to(position)
            #s_label = MathTex(r"s", font_size=24).move_to(position)
            site_group.add(p_top, p_bottom, s_orbital)
            return site_group, s_orbital, p_top, p_bottom
        all_sites = VGroup()
        site_data = []
        for pos in site_positions:
            site_group, s_orb, p_top, p_bottom = create_site(pos)
            all_sites.add(site_group)
            site_data.append({
                's': s_orb.get_center(),
                's-11': s_orb.point_at_angle(2 * PI / 3),
                'p_top-1': p_top.point_at_angle(PI / 3),
                'p_top-11': p_top.point_at_angle(2 * PI / 3)
            })
        # Horizontale Linie durch die Sites
        line_2orb = Line(LEFT * 5.2, RIGHT * 5.2, color=GRAY)
        self.add(line_2orb)
        self.add(all_sites)
        # Nummerierung unter den Sites
        site_labels = VGroup()
        for i, pos in enumerate(site_positions):
            label = Text(str(i + 1), font_size=24).next_to(pos, DOWN * 3)
            site_labels.add(label)
        self.add(site_labels)
        self.wait(0.5)
        # Hilfsfunktion Hopping-Animation
        def animate_hop(start_pos, end_pos, label_text, color=WHITE):
            arrow = ArcBetweenPoints(start_pos, end_pos, angle=-1.5, color=color)
            arrow.add_tip(tip_length=0.15)
            t_label = MathTex(label_text, font_size=30).next_to(arrow, UP, buff=0.2)
            arrow_group = VGroup(arrow, t_label)
            self.play(Create(arrow_group), run_time=0.4)
            self.wait(1.2)  # Länger sichtbar
            self.play(FadeOut(arrow_group), run_time=0.4)
        # Execute Hopping
        animate_hop(site_data[0]['s'], site_data[1]['s-11'], r"\mathbf{t_{ss}", color=TEAL)
        animate_hop(site_data[0]['p_top-1'], site_data[1]['p_top-11'], r"\mathbf{t_{pp}", color=PINK)
        animate_hop(site_data[1]['s'], site_data[2]['p_top-11'], r"\mathbf{t_{sp}")
        animate_hop(site_data[1]['p_top-1'], site_data[2]['s-11'], r"\mathbf{t_{ps}")
        self.wait(1)
        self.play(FadeOut(*self.mobjects))  # Übergang
        BandHybridization(self)




