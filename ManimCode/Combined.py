from manim import *

class Explain(Scene):
    def hop_particle(self, particle, start, end):
        for x in range(start, end):
            arrow = ArcBetweenPoints(RIGHT * x, RIGHT * (x + 1) + UP * 0.1, angle=-2, color=TEAL)
            arrow.add_tip(tip_length=0.15)
            t = Text("t", font_size=24).next_to(arrow, UP * 0.15)
            arrow_group = VGroup(arrow, t)
            self.play(Create(arrow_group), run_time=0.4)
            self.play(FadeOut(particle), run_time=0.3)
            particle.move_to(RIGHT * (x + 1))
            self.play(FadeIn(particle), run_time=0.3)
            self.play(FadeOut(arrow_group), run_time=0.4)

    def construct(self):
        # Horizontale Linie
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
        # Hopping innerhalb der Kette: 3 -> 4 -> 5
        self.hop_particle(particle, 0, 2)
        # Periodischer Übergang: 5 -> 0
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
        # Alte periodische Darstellung ausblenden
        self.play(FadeOut(particle), FadeOut(particle_left), FadeOut(left_0_label), FadeOut(right_0_label), FadeOut(a_lines), FadeOut(a_label), FadeOut(j_labels), FadeOut(boundaries), FadeOut(left_0), FadeOut(right_0), run_time=0.5)
        # Nur Sites 1-5 behalten
        self.wait(0.5)
        # Open system
        explanation = Text("Open system", font_size=30).to_edge(UP)
        self.play(FadeIn(explanation), run_time=0.5)
        self.wait(1)
        # Pump
        pump_arrow = Arrow(LEFT * 3.5, LEFT * 2.1, color=BLUE, buff=0)
        pump_label = Text("Pump", font_size=24).next_to(pump_arrow, UP * 0.2)
        self.play(Create(pump_arrow), FadeIn(pump_label), run_time=0.5)
        # Teilchen wird auf Site 1 gepumpt
        particle = Dot(LEFT * 2, radius=0.12, color=RED_A)
        self.play(FadeIn(particle), run_time=0.3)
        self.play(FadeOut(pump_arrow), FadeOut(pump_label), run_time=0.3)
        # Hopping durch das offene System
        self.hop_particle(particle, -2, 2)
        # Drain
        drain_arrow = Arrow(RIGHT * 2.1, RIGHT * 3.5, color=BLUE, buff=0)
        drain_label = Text("Drain", font_size=24).next_to(drain_arrow, UP * 0.2)
        self.play(Create(drain_arrow), FadeIn(drain_label), run_time=0.5)
        self.play(FadeOut(particle), run_time=0.5)
        self.play(FadeOut(drain_arrow), FadeOut(drain_label), run_time=0.3)
        self.wait(1)
