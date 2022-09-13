# Dance to import our logit model
import os
import sys
from turtle import fillcolor

module_path = os.path.abspath(os.path.join("."))
sys.path.append(module_path)

from voronoi_processing import get_polygons

from manimlib import *
import matplotlib.pyplot as plt
import numpy as np


class Labeling(Scene):
    demo_data = np.load("data/demo_data.npy")
    demo_cluster_history = np.load("data/demo1_cluster_history.npy")
    x_range = [0, 5]
    y_range = [0, 5]
    cluster_cmap = plt.get_cmap("rainbow")

    def construct(self):
        self.construct_graph()
        self.animate()

    def animate(self):
        self.play(Write(self.graph))
        self.play(LaggedStart(*[FadeIn(dot) for dot in self.dots]))
        self.animate_polygons()

    def animate_polygons(self):
        """Animate the centroid Voronoi as they converge."""

        def get_polygon_points(cluster_index):
            polygons = get_polygons(
                self.demo_cluster_history[cluster_index], self.x_range, self.y_range
            )
            polygon_points = [
                self.graph.coords_to_point(*polygons[i].T) for i in range(len(polygons))
            ]
            return polygon_points

        # First, draw the voronoi polygons in
        polygons = []
        polygon_points = get_polygon_points(0)
        for i in range(len(polygon_points)):
            color = rgb_to_hex(self.cluster_cmap(i / len(polygon_points))[:3])
            p = Polygon(
                *polygon_points[i], fill_color=color, fill_opacity=0.5, stroke_width=4
            )
            self.bring_to_back(p)
            self.play(Write(p))
            self.play(p.animate.set_stroke(width=2))
            polygons.append(p)

        for i in range(1, self.demo_cluster_history.shape[0]):
            self.remove(*polygons)
            polygons = []
            polygon_points = get_polygon_points(i)
            for j in range(len(polygon_points)):
                color = rgb_to_hex(self.cluster_cmap(j / len(polygon_points))[:3])
                p = Polygon(
                    *polygon_points[j],
                    fill_color=color,
                    fill_opacity=0.5,
                    stroke_width=2,
                )
                polygons.append(p)
            self.add(*polygons)
            self.bring_to_back(*polygons)
            self.wait(1 / 30)

    def construct_graph(self):
        """Construct the main axes and dots for the animation."""
        self.graph_group = VGroup()

        self.graph = Axes(self.x_range, self.y_range)
        self.graph.add_coordinate_labels(font_size=16, num_decimal_places=0)

        point_coords = self.graph.coords_to_point(*self.demo_data.T)
        self.dots = [
            Dot(point_coords[i], radius=0.05) for i in range(point_coords.shape[0])
        ]
        self.graph_group.add(self.graph, *self.dots)


class LabelUpdates(Scene):
    def construct(self):
        self.construct_left_panel()

        self.animate()
        return

    def animate(self):
        self.play(Write(self.title))
        self.play(Write(self.label_eq[:]))

        self.play(
            TransformMatchingTex(self.label_eq[:1].copy(), self.eq_breakdown[0:2]),
            self.label_eq[0].animate.set_color(RED),
        )
        self.play(
            TransformMatchingTex(self.label_eq[3:4].copy(), self.eq_breakdown[2:4]),
            self.label_eq[4].animate.set_color(BLUE),
        )
        self.play(
            TransformMatchingTex(self.label_eq[6:7].copy(), self.eq_breakdown[4:6]),
            self.label_eq[6].animate.set_color(ORANGE),
        )
        self.play(
            TransformMatchingTex(self.label_eq[2:3].copy(), self.eq_breakdown[6:8]),
            self.label_eq[2].animate.set_color(PURPLE),
        )

        return

    def construct_left_panel(self):
        self.title = Text("K-Means Clustering:\n" "Label Assignment")

        to_isolate = ["{c}", "x", "\\mu", "\\text{argmin}"]
        self.label_eq = Tex(
            "{c}_{n} = \\text{argmin}_{k} \\vert\\vert x_{n} - \\mu_{k} \\vert\\vert^2",
            isolate=[*to_isolate],
        )

        to_isolate = ["{c}", "x", "\\mu", "\\text{argmin}"]
        self.eq_breakdown = Tex(
            "{c}",
            "&\\text{: }\\parbox{3.5cm}{\\raggedright Vector containing labels for input data}",
            "\\\\",
            "x",
            "&\\text{: Input data}",
            "\\\\",
            "\\mu",
            "&\\text{: Cluster centers}",
            "\\\\",
            "\\text{argmin}",
            "&\\text{: }\\parbox{3.5cm}{\\raggedright Reports the index of the cloest cluster center}\\\\",
            # isolate=[*to_isolate]
        )
        # self.eq_breakdown.set_color_by_tex_to_color_map(
        #     {
        #         "{c}": RED,
        #         "x": BLUE,
        #         "\\mu": ORANGE,
        #         "\\text{argmin}": PURPLE
        #     }
        # )
        self.eq_breakdown[0].set_color(RED)
        self.eq_breakdown[2].set_color(BLUE)
        self.eq_breakdown[4].set_color(ORANGE)
        self.eq_breakdown[6].set_color(PURPLE)

        self.left_group = self.group(
            self.title, self.label_eq, self.eq_breakdown, buff=0.5, edge=UL
        )

        return

    def group(
        self,
        *objects,
        arrangement=DOWN,
        buff=0.5,
        edge=None,
    ):
        group = VGroup(*objects)
        group.arrange(arrangement, buff=buff)

        if edge is not None:
            group.to_edge(edge)

        return group


class FunctionDifferentiation(Scene):
    eq_scale = 0.8

    def construct(self):
        self.construct_group_one()
        self.construct_group_four()  # must happen early for location placement
        self.construct_group_two()
        self.construct_group_three()
        self.construct_group_five()
        self.construct_group_six()
        self.construct_group_seven()
        self.construct_final_text()

        self.animate()
        return

    def animate(self):
        # Write card one
        self.play(Write(self.title))
        self.wait(0.1)
        self.play(Write(self.derivative1[0:1]))
        self.play(Write(self.derivative1[1:-1]))
        self.play(Write(self.intro_text_group))
        self.wait(2)
        self.play(FadeOut(self.intro_text_group))
        self.wait(0.5)

        # Write card two
        self.play(ReplacementTransform(self.derivative1[2:-1], self.derivative2))
        self.wait(0.5)

        # Write card three
        self.play(ReplacementTransform(self.derivative2, self.derivative3))
        self.wait(0.5)

        # Write card four
        # Update with replacement card four
        self.remove(self.derivative1, self.derivative2, self.derivative3)
        self.add(self.derivative4[:12])

        self.play(Write(self.k_sum_group), self.derivative4[9].animate.set_color(RED))
        self.wait(2.5)

        self.play(
            TransformMatchingTex(
                self.derivative4[7:12].copy(), self.derivative4[12:16]
            ),
        )
        self.wait(0.5)

        self.play(Write(self.chain_rule_group))
        self.wait(1.5)

        self.play(
            TransformMatchingTex(
                self.derivative4[12:15].copy(), self.derivative4[16:19]
            )
        )
        self.wait(0.5)

        self.play(
            TransformMatchingTex(
                self.derivative4[15:16].copy(), self.derivative4[19:23]
            )
        )
        self.wait(0.5)

        self.play(
            TransformMatchingTex(self.derivative4[15:16].copy(), self.derivative4[23:])
        )
        self.wait(1)

        # Card five
        self.play(TransformMatchingTex(self.derivative4[17:], self.derivative5))
        self.wait(0.5)

        # Card six
        self.remove(self.derivative4[16:], self.derivative5)
        self.add(self.derivative6[:1])
        self.play(TransformMatchingTex(self.derivative5, self.derivative6[1:]))
        self.wait(0.5)

        self.play(
            FadeOut(self.derivative4[:17]),
            FadeOut(self.k_sum_group),
            FadeOut(self.chain_rule_group),
        )
        self.remove(self.derivative4)
        self.wait(0.5)

        # Group 7
        # line 1
        dv6_reposition = self.minimization[0:10].get_left()
        dv6_reposition[0] += self.derivative6.get_width() / 2
        self.play(self.derivative6.animate.move_to(dv6_reposition))
        self.add(self.minimization[0:10])
        self.remove(self.derivative6)
        self.wait(0.5)

        self.play(Write(self.minimization_text))
        self.wait(2)

        # line2
        self.play(
            FadeIn(self.minimization[10]),
            TransformMatchingTex(
                self.minimization[1:10].copy(), self.minimization[11:20]
            ),
        )
        self.wait(0.5)
        self.play(FadeOut(self.minimization_text))
        self.wait(0.5)

        # line 2.2
        self.play(
            TransformMatchingTex(self.minimization[10:20], self.minimization[20:29])
        )
        self.wait(0.5)

        # line 3
        # animate distribution of A_nk
        self.play(
            TransformMatchingTex(
                self.minimization[20:23].copy(), self.minimization[29:32]
            )
        )
        self.play(
            TransformMatchingTex(
                self.minimization[23:24].copy(), self.minimization[32:33]
            ),
            TransformMatchingTex(
                self.minimization[25:26].copy(), self.minimization[33:34]
            ),
        )
        self.play(
            FadeIn(self.minimization[34]),
            TransformMatchingTex(
                self.minimization[23:24].copy(), self.minimization[35:36]
            ),
            TransformMatchingTex(
                self.minimization[27:28].copy(), self.minimization[36:37]
            ),
        )
        self.wait(0.5)

        # line 3.2
        self.play(
            TransformMatchingTex(self.minimization[29:37], self.minimization[37:46])
        )
        self.wait(0.5)

        # line 4
        self.play(
            TransformMatchingTex(
                self.minimization[38:42].copy(), self.minimization[49:53]
            )
        )
        self.play(
            TransformMatchingTex(
                self.minimization[42:46].copy(), self.minimization[46:49]
            )
        )
        self.wait(0.5)

        # line 4.2
        self.play(
            TransformMatchingTex(self.minimization[46:53], self.minimization[53:60])
        )
        self.wait(0.5)

        # line 5
        self.play(
            TransformMatchingTex(
                self.minimization[53:54].copy(), self.minimization[60:62]
            )
        )
        self.play(
            TransformMatchingTex(
                self.minimization[57:60].copy(), self.minimization[62:65]
            )
        )
        self.play(
            TransformMatchingTex(
                self.minimization[54:57].copy(), self.minimization[65:]
            ),
        )
        self.wait(0.5)

        # Final
        self.play(
            FadeOut(self.minimization[0:10]),
            FadeOut(self.minimization[20:29]),
            FadeOut(self.minimization[37:46]),
            FadeOut(self.minimization[53:60]),
        )
        self.play(self.minimization[60:].animate.center().shift(UP).scale(1.23))
        self.wait(0.5)
        self.play(
            FadeIn(self.summary_text1[:1]),
            FadeIn(self.summary_text1[2:]),
            FadeIn(self.summary_text2[:1]),
            FadeIn(self.summary_text3[:1]),
            FadeIn(self.summary_text3[2:3]),
            FadeIn(self.summary_text4[:1]),
            FadeIn(self.summary_text4[2:3]),
            # line 1
            TransformMatchingTex(
                self.minimization[60:61].copy(), self.summary_text1[1:2]
            ),
            self.minimization[60].animate.set_color(RED),
            # line 2
            TransformMatchingTex(
                self.minimization[64:65].copy(), self.summary_text2[1:2]
            ),
            # line 4
            self.minimization[64].animate.set_color(BLUE),
            TransformMatchingTex(
                self.minimization[60:61].copy(), self.summary_text3[1:2]
            ),
            # line 4
            TransformMatchingTex(
                self.minimization[63:64].copy(), self.summary_text4[1:2]
            ),
            self.minimization[63].animate.set_color(PURPLE),
            self.minimization[67].animate.set_color(PURPLE),
        )
        self.wait(5)

        # Fade out
        self.play(*[FadeOut(item) for item in self.mobjects])

        return

    def construct_final_text(self):
        to_isolate = ["\\mu_{k}"]
        self.summary_text1 = Tex(
            "\\text{To minimize the cost function } J \\text{, }",
            "\\mu_{k}",
            "\\text{ is updated}",
            isolate=[*to_isolate],
        )
        self.summary_text1.set_color_by_tex_to_color_map({"\\mu_{k}": RED})

        to_isolate = ["x_{n}"]
        self.summary_text2 = Tex(
            "\\text{to the average position of the labeled points }",
            "x_{n}",
            isolate=[*to_isolate],
        )
        self.summary_text2.set_color_by_tex_to_color_map({"x_{n}": BLUE})

        to_isolate = ["\\mu_{k}"]
        self.summary_text3 = Tex(
            "\\text{that are members of cluster }",
            "\\mu_{k}",
            "\\text{.}",
            isolate=[*to_isolate],
        )
        self.summary_text3.set_color_by_tex_to_color_map({"\\mu_{k}": RED})
        to_isolate = ["A_{nk}"]
        self.summary_text4 = Tex(
            "\\text{This is regulated by }",
            "A_{nk}",
            "\\text{.}",
            isolate=[*to_isolate],
        )
        self.summary_text4.set_color_by_tex_to_color_map({"A_{nk}": PURPLE})

        self.summary_group = VGroup(
            self.summary_text1,
            self.summary_text2,
            self.summary_text3,
            self.summary_text4,
        )
        self.summary_group.arrange(DOWN, buff=0.1)
        self.summary_group.shift(DOWN * 1.5)

        return

    def construct_group_seven(self):
        to_isolate = [
            "x_{n}",
            "\\mu_{k}",
            "\\sum_{n=1}^{N}",
            "A_{nk}",
            "\\frac{dJ}{d\\mu_{k}}",
            "0",
            "=",
        ]
        self.minimization = Tex(
            "\\frac{dJ}{d\\mu_{k}} &= -2 \\sum_{n=1}^{N} A_{nk}(x_{n} - \\mu_{k})\\\\",
            "0 &= -2 \\sum_{n=1}^{N} A_{nk}(x_{n} - \\mu_{k})\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}(x_{n} - \\mu_{k})\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}x_{n} - A_{nk}\\mu_{k}\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}x_{n} - \\sum_{n=1}^{N} A_{nk} \\mu_{k}\\\\",
            "\\sum_{n=1}^{N} A_{nk} \\mu_{k} &= \\sum_{n=1}^{N} A_{nk}x_{n}\\\\",
            "\\mu_{k} \\sum_{n=1}^{N} A_{nk} &= \\sum_{n=1}^{N} A_{nk}x_{n}\\\\",
            "\\mu_{k} &= {\\sum_{n=1}^{N} A_{nk}x_{n} \\over \\sum_{n=1}^{N} A_{nk}}",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        self.minimization_group = self.group(self.title, self.minimization, buffer=0.2)

        self.minimization_text = Tex(
            "\\text{To minimize } J \\text{ w.r.t. } \\mu_{k} \\text{, set } \\frac{dJ}{d\\mu_{k}} = 0"
        ).scale(0.8)
        self.minimization_text.shift(DOWN * 0.5)

        # arrange lines
        eq2_center = self.minimization[10:20].get_center()
        eq3_center = self.minimization[20:29].get_center()
        eq4_center = self.minimization[29:37].get_center()
        eq5_center = self.minimization[37:46].get_center()
        eq6_center = self.minimization[46:53].get_center()
        eq7_center = self.minimization[53:60].get_center()
        eq8_center = self.minimization[60:].get_center()

        # move appropriate lines
        eq3_pos = eq2_center.copy()
        eq3_pos[0] = eq3_center[0]
        self.minimization[20:29].move_to(eq3_pos)

        eq4_pos = eq3_center.copy()
        eq4_pos[0] = eq4_center[0]
        self.minimization[29:37].move_to(eq4_pos)

        eq5_pos = eq4_pos.copy()
        eq5_pos[0] = eq5_center[0]
        self.minimization[37:46].move_to(eq5_pos)

        eq6_pos = eq4_center.copy()
        eq6_pos[0] = eq6_center[0]
        self.minimization[46:53].move_to(eq6_pos)

        eq7_pos = eq6_pos.copy()
        eq7_pos[0] = eq7_center[0]
        self.minimization[53:60].move_to(eq7_pos)

        eq8_pos = eq5_center.copy()
        eq8_pos[0] = eq8_center[0]
        self.minimization[60:].move_to(eq8_pos)

        return

    def construct_group_six(self):
        to_isolate = ["{x_{n} - \\mu_{k}}", "A_{nk}", "\\sum_{n=1}^{N}", "-2"]
        self.derivative6 = Tex(
            "\\frac{dJ}{d\\mu_{k}} = -2 \\sum_{n=1}^{N} A_{nk}({x_{n} - \\mu_{k}})",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        position = self.derivative4[16:].get_left()
        position[0] += self.derivative6.get_width() / 2
        self.derivative6.move_to(position)

        return

    def construct_group_five(self):
        to_isolate = ["{x_{n} - \\mu_{k}}", "A_{nk}", "\\sum_{n=1}^{N}", "-2"]
        self.derivative5 = Tex(
            "\\sum_{n=1}^{N} A_{nk} -2({x_{n} - \\mu_{k}})", isolate=[*to_isolate]
        ).scale(self.eq_scale)

        position = self.derivative4[17:].get_left()
        position[0] += self.derivative5.get_width() / 2
        self.derivative5.move_to(position)

        return

    def construct_group_four(self):
        to_isolate = [
            "\\sum_{n=1}^{N}",
            "\\sum_{k=1}^{K}",
            "A_{nk}",
            "({x_{n} - \\mu_{k}})^2",
            "{x_{n} - \\mu_{k}}",
        ]
        self.derivative4 = Tex(
            "J & = \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk} ||{x_{n} - \mu_{k}}||^2 \\\\",
            "\\frac{dJ}{d\mu_{k}} &= \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk}",
            "({x_{n} - \\mu_{k}})^2 \\\\",
            "\\frac{dJ}{d\\mu_{k}} &= \\sum_{n=1}^{N} A_{nk}",
            "({x_{n} - \\mu_{k}})^2\\\\",
            "\\frac{dJ}{d\mu_{k}} &= ",
            "\\sum_{n=1}^{N} A_{nk}",
            "2",
            "({x_{n} - \\mu_{k}})",
            "\\cdot (0 - 1)",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        self.group_four = self.group(self.title, self.derivative4)

        self.k_sum_group = VGroup()
        self.k_sum_text = Tex("\\text{where } k \\neq \\text{cluster \\#:}").scale(0.65)
        self.k_sum_eq = Tex("{dJ \\over d \\mu_{k}}(x_{n} - \\mu_{k}) = 0").scale(0.65)
        self.k_sum_group.add(self.k_sum_text, self.k_sum_eq)
        self.k_sum_group.arrange(DOWN, buff=0.1)
        self.k_sum_group.move_to(self.derivative4[11].get_right())
        self.k_sum_group.to_edge(RIGHT)

        self.chain_rule_group = VGroup()
        self.chain_rule_text = Tex("\\text{Apply chain rule to:}").scale(0.65)
        self.chain_rule_eq = Tex("(x_{n}-\\mu{k})^2").scale(0.65)
        self.chain_rule_group.add(self.chain_rule_text, self.chain_rule_eq)
        self.chain_rule_group.arrange(DOWN, buff=0.1)
        self.chain_rule_group.move_to(self.derivative4[12].get_right())
        self.chain_rule_group.to_edge(RIGHT)

        return

    def construct_group_three(self):
        to_isolate = ["{{x_{n} - \\mu_{k}}}"]
        self.derivative3 = Tex("({{x_{n} - \mu_{k}}})^2", isolate=[*to_isolate]).scale(
            self.eq_scale
        )

        position = self.derivative4[11].get_left()
        position[0] += self.derivative3.get_width() / 2
        self.derivative3.move_to(position)
        return

    def construct_group_two(self):
        to_isolate = ["{{x_{n} - \mu_{k}}}"]
        self.derivative2 = Tex(
            "((({{x_{n} - \\mu_{k}}})^2)^{1/2})^2", isolate=[*to_isolate]
        ).scale(self.eq_scale)

        position = self.derivative4[11].get_left()
        position[0] += self.derivative2.get_width() / 2
        self.derivative2.move_to(position)
        return

    def construct_group_one(self):

        self.title = Title("K-Means Function Differentiation")

        to_isolate = ["{{x_{n} - \mu_{k}}}"]
        self.derivative1 = Tex(
            "J & = \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk} ||{x_{n} - \mu_{k}}||^2 \\\\",
            "\\frac{dJ}{d\\mu_{k}} &= \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk}",
            "||{{x_{n} - \\mu_{k}}}||^2 \\\\",
            # hidden line for alignment with later equation
            "\\frac{dJ}{d\mu_{i}} &= \\sum_{n=1}^{N} A_{nk} 2 (x_{n} - \\mu_{k}) \\cdot (0 - 1)",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)
        self.group_one = self.group(self.title, self.derivative1)

        self.intro_text1 = Tex(
            "\\text{To minimize } J \\text{ w.r.t. } \\mu_{k} \\text{,}"
        ).scale(0.8)
        self.intro_text2 = Tex(
            "\\text{first differentiate } J \\text{ w.r.t } \\mu_{k}"
        ).scale(0.8)
        self.intro_text_group = VGroup(self.intro_text1, self.intro_text2)
        self.intro_text_group.arrange(DOWN, buff=0.1)
        self.intro_text_group.center().shift(DOWN * 1.2)
        return

    def group(self, *members, buffer=0.5):
        group = VGroup()
        group.add(*members)
        group.arrange(DOWN, buff=buffer)
        group.to_edge(UP)
        return group


class KMeansIntro(Scene):
    def construct(self):
        self.construct_card_one()  # K-means definition
        self.construct_card_two()  # K-means equation
        self.construct_card_three()  # K-means optimization

        self.animate()
        return

    def animate(self):
        self.play(Write(self.title))
        self.wait(0.25)

        # Card one: K-means definition
        self.play(Write(self.definition[:]))
        self.wait(1.5)
        self.play(Write(self.k_definition[:]))
        self.wait(2)
        self.play(
            TransformMatchingTex(self.k_definition[1:2].copy(), self.k_n_eq[0:1]),
            FadeIn(self.k_n_eq[1:2]),
            TransformMatchingTex(self.k_definition[3:4].copy(), self.k_n_eq[2:]),
        )
        self.wait(1.5)
        self.play(FadeOut(self.card_one_group))

        # Card two: K-means equation
        self.play(Write(self.k_eq_definition))
        self.wait(1)
        self.play(Write(self.k_means_eq[:]))
        self.wait(1)
        self.play(Write(self.where))
        self.wait(1)

        self.play(
            self.k_means_eq.animate.set_color_by_tex_to_color_map(
                {"||x_{n} - \mu_{k}||^2": BLUE}
            ),
            TransformMatchingTex(self.k_means_eq[2:3].copy(), self.norm_eq),
            Write(self.norm_def[:]),
        )
        self.wait(3)

        self.play(
            FadeOut(self.norm_eq),
            FadeOut(self.norm_def),
            self.k_means_eq.animate.set_color_by_tex_to_color_map(
                {"||x_{n} - \mu_{k}||^2": WHITE}
            ),
        )
        self.card_two_norm_A_group_update()

        self.play(
            self.k_means_eq.animate.set_color_by_tex_to_color_map({"A_{nk}": RED}),
            TransformMatchingTex(self.k_means_eq[1:2].copy(), self.A_def),
        )
        self.wait(0.5)
        self.play(
            Write(self.kron_1[:]),
        )
        self.wait(2)
        self.play(
            Write(self.kron_2[:]),
        )
        self.wait(2)

        self.play(
            self.kron_1.animate.set_color_by_tex_to_color_map(
                {"{c}": TEAL, "{x}": ORANGE}
            ),
            TransformMatchingTex(self.kron_1[1:2].copy(), self.cn_def[0:1]),
        )
        self.play(
            Write(self.cn_def[1:2]),
        )
        self.play(
            TransformMatchingTex(self.kron_1[4:5].copy(), self.cn_def[2:]),
        )
        self.wait(3)

        self.play(FadeOut(self.card_two_group[1:]))

        # Card three: K-means algorithm
        self.play(Write(self.algorithm_definition[:]))
        self.wait(2)
        self.play(Write(self.label_group[:]))
        self.wait(1.5)
        self.play(Write(self.cluster_group[:]))
        self.wait(1.5)
        self.play(Write(self.final_update[:]))
        self.wait(2)
        self.play(Write(self.arc1))
        self.play(Write(self.arc2))
        self.play(Indicate(self.arc1))
        self.play(Indicate(self.arc2))
        self.play(Indicate(self.arc1))
        self.play(Indicate(self.arc2))
        self.play(FadeOut(self.arc1), FadeOut(self.arc2))

        # Fade out
        self.play(*[FadeOut(item) for item in self.mobjects])
        return

    def construct_card_three(self):
        self.card_three_group = VGroup()

        # Card text
        to_isolate = ["two-step"]
        self.algorithm_definition = TexText(
            """
            After cluster initialization, the k-means function is\\\\
            minimized with an iterative two-step algorithm:
            """,
            isolate=[*to_isolate],
        ).scale(0.9)

        # Card equation groups
        # Update label group
        self.label_group = VGroup()
        self.label_number = Tex("(1)").set_color(BLUE)
        self.label_text = Tex("\\text{Update data labels: }")
        self.label_eq = Tex("{c_{n}}", "= argmin_{k}||{x_{n}} - {\mu_{k}}||^2")

        self.label_group.add(
            self.label_number,
            self.label_text,
            self.label_eq,
        )
        self.label_group.arrange(RIGHT, buff=0.5)
        self.label_group.scale(0.9)

        # Update cluster center group
        self.cluster_group = VGroup()
        self.cluster_number = Tex("(2)").set_color(RED)
        self.cluster_text = Tex("\\text{Update cluster center positions: }")
        self.cluster_eq = Tex(
            """
            \\mu_{k} = 
                {
                    \sum_{n=1}^{N} A_{nk} x_{n} 
                    \\over \sum_{n=1}^{N} A_{nk}
                }
            """
        )

        self.cluster_group.add(self.cluster_number, self.cluster_text, self.cluster_eq)
        self.cluster_group.arrange(RIGHT, buff=0.5)
        self.cluster_group.scale(0.9)

        to_isolate = ["one", "two"]
        self.final_update = TexText(
            "Steps one and two are repeated until the cluster centers\\\\",
            "stabilize (converge) at a local optimum.",
            isolate=[*to_isolate],
        ).scale(0.9)
        self.final_update.set_color_by_tex_to_color_map({"one": BLUE, "two": RED})

        self.card_three_group.add(
            self.title,
            self.algorithm_definition,
            self.label_group,
            self.cluster_group,
            self.final_update,
        )
        self.card_three_group.arrange(DOWN, buff=0.5)
        self.card_three_group.to_edge(UP)

        arc1_start = self.label_eq.get_right()
        arc1_start[0] = self.cluster_eq.get_right()[0]
        self.arc1 = CurvedArrow(arc1_start, self.cluster_eq.get_right(), angle=-TAU / 4)
        self.arc1.shift(RIGHT * 0.3)
        self.arc1.shift(DOWN * 0.1)

        arc2_end = self.label_number.get_left()
        arc2_end[0] += self.cluster_number.get_left()[0] * 0.01
        self.arc2 = CurvedArrow(
            self.cluster_number.get_left(), arc2_end, angle=-TAU / 4
        )
        self.arc2.shift(LEFT * 0.3)

    def card_two_norm_A_group_update(self):
        self.card_two_group = VGroup()
        self.card_two_group.add(
            self.title,
            self.k_eq_definition,
            self.k_means_eq,
            self.where,
            self.A_def_group,
            self.cn_def,
        )
        self.card_two_group.arrange(DOWN, buff=0.5)
        self.card_two_group.to_edge(UP)
        return

    def construct_card_two(self):
        self.card_two_group = VGroup()

        # Definition
        self.k_eq_definition = Text(
            "K-means clustering aims to minimize the following function: \n"
        )

        # Equation
        to_isolate = ["A_{nk}", "||x_{n} - \mu_{k}||^2"]
        self.k_means_eq = Tex(
            "J = \sum_{n=1}^{N} \sum_{k=1}^{K}",
            "A_{nk}",
            "||x_{n} - \mu_{k}||^2",
            isolate=[*to_isolate],
        )

        self.where = Text("where").scale(0.75)

        # Norm definition
        to_isolate = ["||x_{n} - \mu_{k}||^2"]
        self.norm_eq = Tex("||x_{n} - \mu_{k}||^2", isolate=[*to_isolate])
        self.norm_eq.set_color_by_tex_to_color_map({"||x_{n} - \mu_{k}||^2": BLUE})

        to_isolate = ["${x_{n}}$", "${\mu_{k}}$"]
        self.norm_def = TexText(
            "Is the squared distance between data point ${x_{n}}$ \\\\"
            "and cluster center ${\mu_{k}}$",
            isolate=[*to_isolate],
        )

        # Kronecker's delta
        # Manim will not let you isolate within cases :) my workaround
        self.A_def_group = VGroup()

        to_isolate = ["A_{nk}"]
        self.A_def = Tex(
            """
            A_{nk} = 
                \\begin{cases} 
                    & \\\\ 
                    &
                \\end{cases}
            """,
            isolate=[*to_isolate],
        )
        self.A_def.set_color_by_tex_to_color_map({"A_{nk}": RED})

        self.kron_group = VGroup()
        to_isolate = ["{c}", "{x}"]
        self.kron_1 = Tex(
            "1 \\text{ if }",
            "{c}_{n}",
            "= k \\text{ (i.e., }",
            "{x}_{n}",
            "\\text{ is a member of cluster } \mu_{k} \\text{)}",
            isolate=[*to_isolate],
        )
        self.kron_2 = Tex("0 \\text{ otherwise }")

        self.kron_group.add(self.kron_1, self.kron_2)
        self.kron_group.arrange_in_grid(2, 1, 0.1, aligned_edge=LEFT)

        self.A_def_group.add(self.A_def, self.kron_group)
        self.A_def_group.arrange(RIGHT, buff=0.1)

        to_isolate = ["{c}", "{x}"]
        self.cn_def = Tex(
            "{c}",
            "\\text{ is a vector containing the cluster membership of all }",
            "{x}",
            isolate=[*to_isolate],
        )
        self.cn_def.set_color_by_tex_to_color_map({"{c}": TEAL, "{x}": ORANGE}),

        self.card_two_group.add(
            self.title,
            self.k_eq_definition,
            self.k_means_eq,
            self.where,
            self.norm_eq,
            self.norm_def,
        )
        self.card_two_group.arrange(DOWN, buff=0.5)
        self.card_two_group.to_edge(UP)
        return

    def construct_card_one(self):
        self.title = Title("K-Means Clustering")
        self.title.to_edge(UP)

        self.card_one_group = VGroup()

        to_isolate = ["${k}$"]
        self.definition = TexText(
            """
            K-means clustering is an unsupervised algorithm used\n
            to cluster input data into ${k}$ groups.
            """,
            isolate=[*to_isolate],
        )
        self.definition.set_color_by_tex_to_color_map({"${k}$": BLUE})

        to_isolate = ["{k}", "{n}"]

        to_isolate = ["${k}$", "${n}$"]
        self.k_definition = TexText(
            "The number of clusters (",
            "${k}$",
            ") is user-defined and must not \\\\"
            "exceed the number of input data points ("
            "${n}$",
            ").",
            isolate=[*to_isolate],
        )
        self.k_definition.set_color_by_tex_to_color_map({"{k}": BLUE, "{n}": ORANGE})

        to_isolate = ["${k}$", "${n}$"]
        self.k_n_eq = TexText("${k}$", "$\leq$", "${n}$", isolate=[*to_isolate])
        self.k_n_eq.set_color_by_tex_to_color_map({"${k}$": BLUE, "${n}$": ORANGE})

        self.card_one_group.add(self.definition, self.k_definition, self.k_n_eq)
        self.card_one_group.arrange(DOWN, buff=0.5)

        return
