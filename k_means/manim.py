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
                    stroke_width=2
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


class FunctionDifferentiation(Scene):
    def construct(self):
        
        return
    
    def animate(self):
        
        return
    
    def construct_card(self):
        self.title = Title("K-Means Function Differentiation")
        
        
        return


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
        self.play(FadeIn(self.definition))
        self.wait(1.5)
        self.play(FadeIn(self.k_definition))
        self.wait(2)
        self.play(
            TransformMatchingTex(self.k_definition[1:2].copy(), self.k_n_eq[0:1]),
            FadeIn(self.k_n_eq[1:2]),
            TransformMatchingTex(self.k_definition[3:4].copy(), self.k_n_eq[2:]),
        )
        self.wait(1.5)
        self.play(FadeOut(self.card_one_group))

        # Card two: K-means equation
        self.play(FadeIn(self.k_eq_definition))
        self.wait(1)
        self.play(FadeIn(self.k_means_eq))
        self.wait(1)
        self.play(FadeIn(self.where))
        self.wait(1)

        self.play(
            self.k_means_eq.animate.set_color_by_tex_to_color_map(
                {"||x_{n} - \mu_{k}||^2": BLUE}
            ),
            TransformMatchingTex(self.k_means_eq[2:3].copy(), self.norm_eq),
            FadeIn(self.norm_def),
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
            FadeIn(self.kron_1),
        )
        self.wait(2)
        self.play(
            FadeIn(self.kron_2),
        )
        self.wait(2)

        self.play(
            self.kron_1.animate.set_color_by_tex_to_color_map(
                {"{c}": TEAL, "{x}": ORANGE}
            ),
            TransformMatchingTex(self.kron_1[1:2].copy(), self.cn_def[0:1]),
            FadeIn(self.cn_def[1]),
            TransformMatchingTex(self.kron_1[4:5].copy(), self.cn_def[2:]),
        )
        self.wait(3)

        self.play(FadeOut(self.card_two_group[1:]))

        # Card three: K-means algorithm
        self.play(FadeIn(self.algorithm_definition))
        self.wait(2)
        self.play(FadeIn(self.label_group))
        self.wait(1.5)
        self.play(FadeIn(self.cluster_group))
        self.wait(1.5)
        self.play(FadeIn(self.final_update))
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
        # self.k_definition = Tex(
        #     "\\text{The number of clusters (}",
        #     "{k}",
        #     "\\text{) is user-defined and must not }\\\\"
        #     "\\text{exceed the number of input data points (}"
        #     "{n}",
        #     "\\text{).}",
        #     isolate=[*to_isolate],
        # )
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
