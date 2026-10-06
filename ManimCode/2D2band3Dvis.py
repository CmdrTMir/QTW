from manim import *
import numpy as np

class BandStructure3D(ThreeDScene):
    def construct(self):
        # --- 1. Parameter ---
        t = 1.0
        m_tracker = ValueTracker(-3.0)

        # --- 2. 3D-Koordinatensystem ---
        range_min, range_max = -PI, PI
        padding = 0.6

        axes = ThreeDAxes(
            x_range=[range_min - padding, range_max + padding, PI/2],
            y_range=[range_min - padding, range_max + padding, PI/2],
            z_range=[-10 - padding, 14 + padding, 2],
            x_length=8,
            y_length=8,
            z_length=6,
            axis_config={"include_tip": False, "color": WHITE}
        )

        def grid_func(u, v):
            return axes.c2p(u, v, 0)
        grid_plane = Surface(
            grid_func,
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(15, 15),
            fill_opacity=0.0,
            stroke_color=GREY_C,
            stroke_width=1.5,
            stroke_opacity=0.5
        )
        #self.play(Create(grid_plane))

        ### Ich weiß einfach nicht wie ich die Achsen richtig bekomme...

        # range_min, range_max = -PI, PI
        # padding = 0.6
        #
        # axes = ThreeDAxes(
        #     x_range=[range_min - padding, range_max + padding, PI/2],
        #     y_range=[range_min - padding, range_max + padding, PI/2],
        #     z_range=[-10 - padding, 14 + padding, 2],
        #     x_length=8,
        #     y_length=8,
        #     z_length=6,
        #     axis_config={"include_tip": False, "color": WHITE}
        # )
        #
        # x_label = MathTex("k_x", font_size=30).next_to(axes.x_axis, RIGHT)
        # y_label = MathTex("k_y", font_size=30).next_to(axes.y_axis, UP)
        # z_label = MathTex("E", font_size=30).next_to(axes.z_axis, UP)
        # axes_labels = VGroup(x_label, y_label, z_label)
        #
        # x_ticks = VGroup()
        # x_labels = VGroup()
        # for val, label in [(-PI, ""), (-PI/2, ""), (0, ""), (PI/2, ""), (PI, "")]:
        #     tick = Line(axes.c2p(val, 0), axes.c2p(val, -0.15), color=WHITE)
        #     lbl = MathTex(label, font_size=28).next_to(tick, DOWN, buff=0.1)
        #     x_ticks.add(tick)
        #     x_labels.add(lbl)
        #
        # y_ticks = VGroup()
        # y_labels = VGroup()
        # for val, label in [(-PI, ""), (-PI/2, ""), (0, ""), (PI/2, ""), (PI, "")]:
        #     tick = Line(axes.c2p(val, 0), axes.c2p(val, -0.15), color=WHITE)
        #     lbl = MathTex(label, font_size=28).next_to(tick, DOWN, buff=0.1)
        #     y_ticks.add(tick)
        #     y_labels.add(lbl)
        #
        # z_ticks = VGroup()
        # z_labels = VGroup()
        # for val in [-4, 4, 8]:
        #     tick = Line(axes.c2p(0, val), axes.c2p(-0.08, val), color=WHITE)
        #     lbl = MathTex(str(val), font_size=20).next_to(tick, LEFT, buff=0.1)
        #     z_ticks.add(tick)
        #     z_labels.add(lbl)
        #
        # self.add(x_ticks, x_labels, y_ticks, y_labels, z_ticks, z_label)

        # --- 3. Hilfsfunktion zur Erstellung der Oberflächen ---
        def create_surface(is_upper):
            m = m_tracker.get_value()

            def func(kx, ky):
                val = np.sqrt(np.sin(kx)**2 + np.sin(ky)**2 + (m + np.cos(kx) + np.cos(ky))**2)
                z = 2 * t
                if is_upper:
                    z += 2 * t * val
                else:
                    z -= 2 * t * val
                return axes.c2p(kx, ky, z)

            color = PINK if is_upper else TEAL
            return Surface(
                func,
                u_range=[-PI, PI],  # Manims Surface erwartet u_range/v_range als Argumentnamen
                v_range=[-PI, PI],  # auch wenn die Funktion kx, ky verwendet
                resolution=(25, 25),
                fill_color=color,
                fill_opacity=0.85,
                stroke_color=WHITE,
                stroke_width=0.5
            )

        # Oberflächen
        surface_minus = create_surface(False)
        surface_plus = create_surface(True)
        surface_minus.add_updater(lambda m: m.become(create_surface(False)))
        surface_plus.add_updater(lambda m: m.become(create_surface(True)))

        # --- 4. Anzeige der Parameter und der Chern-Zahl ---
        m_label = MathTex(r"m = ", font_size=36)
        m_value = DecimalNumber(-3.00, num_decimal_places=2, font_size=36)
        m_group = VGroup(m_label, m_value).arrange(RIGHT, buff=0.1)
        chern_label = MathTex(r"C = ", font_size=36)
        chern_value = Integer(0, font_size=36)
        chern_group = VGroup(chern_label, chern_value).arrange(RIGHT, buff=0.1)

        def update_m(mob):
            mob.set_value(m_tracker.get_value())
            self.add_fixed_in_frame_mobjects(mob)

        def update_chern(mob):
            m = m_tracker.get_value()
            if m > 2: c = 0
            elif 0 < m < 2: c = -1
            elif -2 < m < 0: c = 1
            else: c = 0
            mob.set_value(c)
            self.add_fixed_in_frame_mobjects(mob)

        chern_value.add_updater(update_chern)
        m_value.add_updater(update_m)
        param_group = VGroup(m_group, chern_group).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        param_group.to_corner(UR)
        self.add_fixed_in_frame_mobjects(param_group)
        self.play(FadeIn(param_group), run_time=1)

        # --- 5. Kamera und Animation ---
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
        #self.play(Create(axes), Write(axes_labels))
        #self.wait(0.5)
        self.play(Create(surface_minus), Create(surface_plus), run_time=2)
        self.wait(0.5)
        self.play(
            m_tracker.animate.set_value(3.0),
            run_time=12,
            rate_func=linear
        )
        self.wait(1)
