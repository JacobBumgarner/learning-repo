# Dance to import our logit model
import os
import sys
from sre_constants import JUMP

module_path = os.path.abspath(os.path.join("."))
sys.path.append(module_path)

from voronoi_processing import get_polygons

from manimlib import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class KMeansAlgo(Scene):
    def construct_parameters(self):
        
        return
    
    def construct(self):
        
        return
    
    def animate(self):
        
        return
    
    


class KMeansInit(Scene):
    def construct_parameters(self):
        self.data = np.load("data/synth_data.npy")
        self.data = np.pad(self.data, ((0, 0), (0, 1)))  # manim needs 3D coordinates
        self.centroids = np.load("data/centroid_history2.npy")[0]
        self.centroids = np.pad(self.centroids, ((0, 0), (0, 1)))
        self.centroid_objects = []

        self.DOT_COLOR = "#a7b8c7"
        self.C1_COLOR = "#00b4eb"
        self.C2_COLOR = "#74feba"
        self.C3_COLOR = "#ffa256"
        self.C4_COLOR = "#ff4d27"
        self.C_COLORS = [self.C1_COLOR, self.C2_COLOR, self.C3_COLOR, self.C4_COLOR]
        self.PX_COLOR = "#DB238B"
        self.DX_COLOR = "#02BDCF"

    def construct(self):
        self.construct_parameters()

        self.construct_left_panel()
        self.construct_dividing_line()
        self.construct_right_panel()
        self.group_page()

        self.animate()
        return

    def animate(self):
        # Create scenel, sett up data
        self.play(LaggedStart(*[Write(self.title[:]), Write(self.title_ul)]))
        self.play(Write(self.dividing_line), Write(self.graph))
        self.play(LaggedStart(*[FadeIn(dot) for dot in self.dots]), run_time=2)

        # Choose first centroid
        self.play(Write(self.step1[:]))
        self.wait(0.5)
        self.play(self.construct_emphasize_centroid(0))
        self.wait(0.25)

        # Describe algorithm to choose other centroids
        self.play(Write(self.step2[:]))

        self.play(Write(self.probability_distribution[:]))
        self.play(
            self.probability_distribution.animate.set_color_by_tex_to_color_map(
                {"D(x_{n})": self.DX_COLOR}
            )
        )
        self.play(
            TransformMatchingTex(self.probability_distribution[1:].copy(), self.dX_def)
        )
        self.wait(1)

        # Construct the axis for the probability distribution
        self.play(Write(self.prob_axis), Write(self.prob_axis_label))

        # Animate the other centroids
        distance_lag_intervals = [0.05, 0.025]
        distribution_lag_intervals = [0.025, 0.01]
        for i in range(0, self.centroids.shape[0] - 1):
            # Either write or emphasize the third step
            if i == 1:
                self.play(Write(self.step3[:]))
            elif i > 1:
                self.play(Indicate(self.step3))
            self.wait(1)

            self.construct_animate_centroid_distance_lines(
                lag_interval=distance_lag_intervals[i > 0]
            )
            self.wait(0.4)

            self.animate_probability_distribution(
                lag_interval=distribution_lag_intervals[i > 0]
            )
            self.wait(0.25)

            self.animate_sort_probability_distribution()
            self.wait(0.5)

            self.animate_probability_selection(i + 1)
            self.wait(0.25)

            self.animate_probability_distriubtion_fade()
            self.wait(0.25)

        # Fade out the probability distribution line
        self.wait(0.75)
        self.play(
            FadeOut(self.prob_axis_group, shift=DOWN),
            # FadeOut(self.prob_axis_group),
            self.graph_group.animate.shift(DOWN * 0.7),
        )
        self.wait(1)

        # Fade everything out
        self.play(*[FadeOut(item) for item in self.mobjects])
        return

    def animate_probability_distriubtion_fade(self):
        self.play(
            *[FadeOut(dot_group) for dot_group in self.dot_groups],
            *[FadeOut(line) for line in self.final_lines],
        )

        return

    def animate_probability_selection(self, centroid_index):

        # Animate the random selection of the new centroid from the generated prob dist
        # First find the index of the centroids from the previously selected data
        dot_orig_index = np.argwhere(
            self.data[:, :2] == self.centroids[centroid_index, :2]
        )[0, 0]

        # Now indicate the centroids and dot_groups until we hit the selected centroid
        self.sorted_dot_indices = np.flip(self.sorted_dot_indices)
        dot_sorted_index = np.argwhere(self.sorted_dot_indices == dot_orig_index)[0, 0]

        group_indications = []
        dot_indications = []
        graph_line_indications = []
        # Construct the animations for all but the final selection
        selection_color = self.C_COLORS[centroid_index]
        for i in range(0, dot_sorted_index + 1):
            if i < dot_sorted_index:
                dot_animation = Indicate(
                    self.dots[self.sorted_dot_indices[i]],
                    color=selection_color,
                    scale_factor=3,
                )
                group_animation = Indicate(
                    self.dot_groups[self.sorted_dot_indices[i]],
                    color=selection_color,
                    scale_factor=1.5 if i < dot_sorted_index else 2.5,
                )

                graph_line_animation = Indicate(
                    self.final_lines[self.sorted_dot_indices[i]],
                    color=selection_color,
                    scale_factor=1,
                )
            else:
                dot_animation = self.construct_emphasize_centroid(centroid_index)
                group_animation = (
                    self.dot_groups[self.sorted_dot_indices[i]]
                    .animate.set_color(selection_color)
                    .scale(1.75)
                )
                graph_line_animation = self.final_lines[
                    self.sorted_dot_indices[i]
                ].animate.set_color(selection_color)

            dot_indications.append(dot_animation)
            group_indications.append(group_animation)
            graph_line_indications.append(graph_line_animation)

        # Play the indications up to the final selection
        self.play(
            LaggedStart(*group_indications, lag_ratio=0.5),
            LaggedStart(*dot_indications, lag_ratio=0.5),
            LaggedStart(*graph_line_indications, lag_ratio=0.5),
        )

        return

    def animate_sort_probability_distribution(self):
        # Sort the probability distribution
        animations = []
        positions = [group.get_bottom() for group in self.dot_groups]
        self.sorted_dot_indices = np.argsort(self.line_lengths[:, 1])

        # Construct the animations
        for i, dot_group in enumerate(self.dot_groups):
            group_sorted_index = np.argwhere(i == self.sorted_dot_indices)[0, 0]
            position = positions[group_sorted_index]
            position[1] += dot_group.get_height() / 2

            animations.append(dot_group.animate.move_to(position))

        self.play(*animations)
        self.add(*[group[1:] for group in self.dot_groups])
        return

    def animate_probability_distribution(self, lag_interval):
        # Generate the probability distribution
        left_pos = self.graph.get_left() - [0.05, 0, 0]
        right_pos = self.graph.get_right() + [0.2, 0, 0]
        placements = np.linspace(left_pos, right_pos, len(self.final_lines))
        placements[:, 1] = self.prob_axis.number_to_point(0)[1]

        # Get the line lengths and convert them into their probabilities
        self.line_lengths = np.array(
            [[0, line.get_length(), 0] for line in self.final_lines]
        )
        self.line_lengths /= np.max(self.line_lengths)

        # Construct the animations
        animations = []
        self.dot_groups = []
        self.new_lines = []
        for i, dot_group in enumerate(zip(self.dots, self.final_lines)):
            dot, line = dot_group

            # dot
            new_dot = dot.copy()
            animations.append(new_dot.animate.move_to(placements[i]))

            # dot label
            label_number = str(i)
            dot_label = Text(f"{label_number}").scale(0.2)
            dot_label.move_to(placements[i] - [0, 0.15, 0])
            animations.append(FadeIn(dot_label))

            # line repositioning
            new_line = line.copy()
            # new_line.set_stroke(width=3)

            end_adjustment = [0, new_dot.get_height() / 2, 0]
            animations.append(
                new_line.animate.put_start_and_end_on(
                    placements[i] + self.line_lengths[i],
                    placements[i] + end_adjustment,
                ).set_stroke(width=3)
            )
            self.dot_groups.append(VGroup(new_line, new_dot, dot_label))

        # Animate it
        self.play(LaggedStart(*animations, lag_ratio=lag_interval))

        return

    def construct_animate_centroid_distance_lines(self, lag_interval):
        # Construct and animate the distance lines between the centroids and input data.
        # Findg the index of the closest centroid to the current data
        distances = cdist(self.data, self.centroids[: len(self.centroid_objects)])
        closest_centroid_indices = np.argmin(distances, axis=1)

        # Construct the lines
        self.all_lines = []
        self.temp_lines = []
        self.final_lines = []
        for i, dot in enumerate(self.dots):
            for j, centroid_object in enumerate(self.centroid_objects):
                line = Line(
                    centroid_object.get_center(),
                    dot.get_center(),
                    color=self.C_COLORS[j],
                    stroke_width=1.5,
                )

                self.all_lines.append(line)
                if j == closest_centroid_indices[i]:
                    self.final_lines.append(line)
                else:
                    self.temp_lines.append(line)

        # Animate the distance line projections to and away from the input data
        self.play(
            LaggedStart(
                *[Write(line) for line in self.all_lines], lag_ratio=lag_interval
            ),
        )

        if self.temp_lines:  # If there are temporary distance lines, uncreate them
            self.play(Indicate(self.dX_def[2], scale_factor=2))
            self.play(
                *[Uncreate(line) for line in self.temp_lines],
            )

        return

    def construct_emphasize_centroid(self, centroid_index):
        # Emphasize the selected data point and convert it into a centroid
        # First, find and remove the dot that will become the new centroid
        old_index = np.argwhere(self.data[:, :2] == self.centroids[centroid_index, :2])[
            0, 0
        ]
        self.data = np.delete(self.data, old_index, axis=0)
        old_dot = self.dots.pop(old_index)

        scaled_centroid = Dot(
            self.graph.coords_to_point(*self.centroids[centroid_index]),
            radius=0.1,
            color=self.C_COLORS[centroid_index],
        )

        self.centroid_objects.append(scaled_centroid)
        self.graph_group.add(scaled_centroid, old_dot)

        animation = LaggedStart(
            Indicate(old_dot, scale_factor=6, color=self.C_COLORS[centroid_index]),
            FadeIn(scaled_centroid),
        )
        return animation

    def group_page(self):
        # Group the left group, line, and right group into a single unit
        self.page_group = VGroup(self.left_group, self.dividing_line, self.right_group)
        self.page_group.arrange(RIGHT, buff=0.3)
        self.page_group.to_edge(LEFT)

        self.right_group.shift(LEFT * 0.15).shift(DOWN * 0.1)
        return

    def construct_probability_axis(self):
        # Construct the axis
        self.prob_axis = NumberLine(
            [0, 1],
            width=1,
            tick_size=0.05,
            line_to_number_direction=LEFT,
            line_to_number_buff=0.1,
        ).rotate(90 * DEGREES, about_point=ORIGIN)
        # Set stroke first to only affect axis and not text width
        self.prob_axis.set_stroke(width=3, color=WHITE)

        position = self.graph.get_left() - [0.3, 0, 0]
        position[1] += (
            self.graph.get_bottom()[1] * 1.65 + self.prob_axis.get_height() / 2
        )

        self.prob_axis.add_numbers([0, 1], font_size=24)
        self.prob_axis.move_to(position)

        # Construct the axis label
        to_isolate = ["P(x_{n})"]
        self.prob_axis_label = Tex(
            "{P(x_{n}) \\over max(P(x_{n}))", isolate=[*to_isolate]
        )
        self.prob_axis_label.set_color_by_tex_to_color_map({"P(x_{n})": self.PX_COLOR})
        self.prob_axis_label.rotate(PI / 2).scale(0.4)
        label_position = self.prob_axis.get_center()
        label_position[0] -= 0.4
        self.prob_axis_label.move_to(label_position)

        self.prob_axis_group = VGroup(self.prob_axis, self.prob_axis_label)

        return

    def construct_graph(self):
        self.graph = Axes([0, 5], [0, 5], width=7.14, height=4.64, tick_size=0.05)
        self.graph.get_axes().set_stroke(width=3, color=WHITE)
        # self.graph.add_coordinate_labels(font_size=28, num_decimal_places=0)

        # update position of dots
        for i, dot in enumerate(self.dots):
            dot.move_to(self.graph.coords_to_point(*self.data[i]))

        # Create graph group
        self.graph_group = VGroup()
        self.graph_group.add(self.graph, *self.dots)

        return

    def construct_dots(self):
        # Convert the input data to dots
        # Keep the radius small
        self.dots = [
            Dot(radius=0.05, color=self.DOT_COLOR) for _ in range(self.data.shape[0])
        ]
        return

    def construct_right_panel(self):
        # Call the construction of the right panel components graph components
        self.construct_dots()
        self.construct_graph()
        self.construct_probability_axis()

        # Group them
        self.right_group = VGroup()
        self.right_group.add(self.graph_group, self.prob_axis_group)
        # self.right_group.arrange(DOWN, buff=0)
        return

    def construct_dividing_line(self):
        line_start = self.left_group.get_top()
        line_start[0] = self.left_group.get_right()[0]
        line_end = self.left_group.get_bottom()
        line_end[0] = self.left_group.get_right()[0]
        self.dividing_line = Line(line_start, line_end)
        return

    def construct_left_panel(self):
        # Organized into three separate Texes for the steps and two texes for the eqs
        self.title1 = Text("KMeans++ Cluster")
        self.title2 = TexText("Initialization $(k = 4)$")
        self.title = VGroup(self.title1, self.title2).arrange(DOWN, buff=0.2)
        self.title_ul = Underline(self.title)
        self.title_group = (
            VGroup(self.title, self.title_ul).arrange(DOWN, buff=0.2).scale(0.75)
        )

        to_isolate = ["${X}$"]
        self.step1 = TexText(
            "1. Choose one center ${c_{1}}$ \\\\",
            "randomly from ${X}$.",
            isolate=[*to_isolate],
        ).scale(0.7)
        self.step1.set_color_by_tex_to_color_map({"${X}$": self.DOT_COLOR})

        to_isolate = ["${X}$", "${P(x_{n})}$"]
        self.step2 = TexText(
            "2. Choose a new center ${c_{i}}$ \\\\",
            "from ${X}$ using probability \\\\",
            "distribution weights ${P(x_{n})}$. \\\\",
            isolate=[*to_isolate],
        ).scale(0.7)
        self.step2.set_color_by_tex_to_color_map(
            {"${X}$": self.DOT_COLOR, "${P(x_{n})}$": self.PX_COLOR}
        )

        self.step3 = TexText(
            "3. Repeat step 2 until ${k}$\\\\",
            "centers have been selected",
            isolate=[*to_isolate],
        ).scale(0.7)

        to_isolate = ["P(x_{n})", "D(x_{n})"]
        self.probability_distribution = Tex(
            "P(x_{n}) = {D(x_{n}) \\over \\sum_{n=1}^{N}D(x_{n})}",
            isolate=[*to_isolate],
        ).scale(0.8)
        self.probability_distribution.set_color_by_tex_to_color_map(
            {"P(x_{n})": self.PX_COLOR}
        )

        to_isolate = ["D(x_{n})", "\\min_{j < i}"]
        self.dX_def = Tex(
            "D(x_{n}) = \\min_{j < i}||x_{n} - c_{j}||^2", isolate=[*to_isolate]
        ).scale(0.8)
        self.dX_def.set_color_by_tex_to_color_map({"D(x_{n})": self.DX_COLOR})

        # Create and organize the group
        self.left_group = VGroup(
            self.title_group,
            self.step1,
            self.step2,
            self.probability_distribution,
            self.dX_def,
            self.step3,
        )
        self.left_group.arrange(DOWN, buff=0.4)
        self.left_group.to_edge(UL)

        return


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
        )
        self.wait(5)

        # Fade out
        self.play(*[FadeOut(item) for item in self.mobjects])

        return

    def construct_final_text(self):
        to_isolate = ["c_{k}"]
        self.summary_text1 = Tex(
            "\\text{To minimize the cost function } J \\text{, }",
            "c_{k}",
            "\\text{ is updated}",
            isolate=[*to_isolate],
        )
        self.summary_text1.set_color_by_tex_to_color_map({"c_{k}": RED})

        to_isolate = ["x_{n}"]
        self.summary_text2 = Tex(
            "\\text{to the average position of the labeled points }",
            "x_{n}",
            isolate=[*to_isolate],
        )
        self.summary_text2.set_color_by_tex_to_color_map({"x_{n}": BLUE})

        to_isolate = ["c_{k}"]
        self.summary_text3 = Tex(
            "\\text{that are members of cluster }",
            "c_{k}",
            "\\text{.}",
            isolate=[*to_isolate],
        )
        self.summary_text3.set_color_by_tex_to_color_map({"c_{k}": RED})

        self.summary_group = VGroup(
            self.summary_text1,
            self.summary_text2,
            self.summary_text3,
        )
        self.summary_group.arrange(DOWN, buff=0.1)
        self.summary_group.shift(DOWN * 1.5)

        return

    def construct_group_seven(self):
        to_isolate = [
            "x_{n}",
            "c_{k}",
            "\\sum_{n=1}^{N}",
            "A_{nk}",
            "\\frac{dJ}{dc_{k}}",
            "0",
            "=",
        ]
        self.minimization = Tex(
            "\\frac{dJ}{dc_{k}} &= -2 \\sum_{n=1}^{N} A_{nk}(x_{n} - c_{k})\\\\",
            "0 &= -2 \\sum_{n=1}^{N} A_{nk}(x_{n} - c_{k})\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}(x_{n} - c_{k})\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}x_{n} - A_{nk}c_{k}\\\\",
            "0 &= \\sum_{n=1}^{N} A_{nk}x_{n} - \\sum_{n=1}^{N} A_{nk} c_{k}\\\\",
            "\\sum_{n=1}^{N} A_{nk} c_{k} &= \\sum_{n=1}^{N} A_{nk}x_{n}\\\\",
            "c_{k} \\sum_{n=1}^{N} A_{nk} &= \\sum_{n=1}^{N} A_{nk}x_{n}\\\\",
            "c_{k} &= {\\sum_{n=1}^{N} A_{nk}x_{n} \\over \\sum_{n=1}^{N} A_{nk}}",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        self.minimization_group = self.group(self.title, self.minimization, buffer=0.2)

        self.minimization_text = Tex(
            "\\text{To minimize } J \\text{ w.r.t. } c_{k} \\text{, set } \\frac{dJ}{dc_{k}} = 0"
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
        to_isolate = ["{x_{n} - c_{k}}", "A_{nk}", "\\sum_{n=1}^{N}", "-2"]
        self.derivative6 = Tex(
            "\\frac{dJ}{dc_{k}} = -2 \\sum_{n=1}^{N} A_{nk}({x_{n} - c_{k}})",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        position = self.derivative4[16:].get_left()
        position[0] += self.derivative6.get_width() / 2
        self.derivative6.move_to(position)

        return

    def construct_group_five(self):
        to_isolate = ["{x_{n} - c_{k}}", "A_{nk}", "\\sum_{n=1}^{N}", "-2"]
        self.derivative5 = Tex(
            "\\sum_{n=1}^{N} A_{nk} -2({x_{n} - c_{k}})", isolate=[*to_isolate]
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
            "({x_{n} - c_{k}})^2",
            "{x_{n} - c_{k}}",
        ]
        self.derivative4 = Tex(
            "J & = \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk} ||{x_{n} - c_{k}}||^2 \\\\",
            "\\frac{dJ}{dc_{k}} &= \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk}",
            "({x_{n} - c_{k}})^2 \\\\",
            "\\frac{dJ}{dc_{k}} &= \\sum_{n=1}^{N} A_{nk}",
            "({x_{n} - c_{k}})^2\\\\",
            "\\frac{dJ}{dc_{k}} &= ",
            "\\sum_{n=1}^{N} A_{nk}",
            "2",
            "({x_{n} - c_{k}})",
            "\\cdot (0 - 1)",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)

        self.group_four = self.group(self.title, self.derivative4)

        self.k_sum_group = VGroup()
        self.k_sum_text = Tex("\\text{where } k \\neq \\text{cluster \\#:}").scale(0.65)
        self.k_sum_eq = Tex("{dJ \\over d c_{k}}(x_{n} - c_{k}) = 0").scale(0.65)
        self.k_sum_group.add(self.k_sum_text, self.k_sum_eq)
        self.k_sum_group.arrange(DOWN, buff=0.1)
        self.k_sum_group.move_to(self.derivative4[11].get_right())
        self.k_sum_group.to_edge(RIGHT)

        self.chain_rule_group = VGroup()
        self.chain_rule_text = Tex("\\text{Apply chain rule to:}").scale(0.65)
        self.chain_rule_eq = Tex("(x_{n}-c{k})^2").scale(0.65)
        self.chain_rule_group.add(self.chain_rule_text, self.chain_rule_eq)
        self.chain_rule_group.arrange(DOWN, buff=0.1)
        self.chain_rule_group.move_to(self.derivative4[12].get_right())
        self.chain_rule_group.to_edge(RIGHT)

        return

    def construct_group_three(self):
        to_isolate = ["{{x_{n} - c_{k}}}"]
        self.derivative3 = Tex("({{x_{n} - c_{k}}})^2", isolate=[*to_isolate]).scale(
            self.eq_scale
        )

        position = self.derivative4[11].get_left()
        position[0] += self.derivative3.get_width() / 2
        self.derivative3.move_to(position)
        return

    def construct_group_two(self):
        to_isolate = ["{{x_{n} - c_{k}}}"]
        self.derivative2 = Tex(
            "((({{x_{n} - c_{k}}})^2)^{1/2})^2", isolate=[*to_isolate]
        ).scale(self.eq_scale)

        position = self.derivative4[11].get_left()
        position[0] += self.derivative2.get_width() / 2
        self.derivative2.move_to(position)
        return

    def construct_group_one(self):

        self.title = Title("K-Means Function Differentiation")

        to_isolate = ["{{x_{n} - c_{k}}}"]
        self.derivative1 = Tex(
            "J & = \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk} ||{x_{n} - c_{k}}||^2 \\\\",
            "\\frac{dJ}{dc_{k}} &= \\sum_{n=1}^{N} \\sum_{k=1}^{K} A_{nk}",
            "||{{x_{n} - c_{k}}}||^2 \\\\",
            # hidden line for alignment with later equation
            "\\frac{dJ}{dc_{i}} &= \\sum_{n=1}^{N} A_{nk} 2 (x_{n} - c_{k}) \\cdot (0 - 1)",
            isolate=[*to_isolate],
        ).scale(self.eq_scale)
        self.group_one = self.group(self.title, self.derivative1)

        self.intro_text1 = Tex(
            "\\text{To minimize } J \\text{ w.r.t. } c_{k} \\text{,}"
        ).scale(0.8)
        self.intro_text2 = Tex(
            "\\text{first differentiate } J \\text{ w.r.t } c_{k}"
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
        self.wait(1.5)
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
                {"||x_{n} - c_{k}||^2": BLUE}
            ),
            TransformMatchingTex(self.k_means_eq[2:3].copy(), self.norm_eq),
            Write(self.norm_def[:]),
        )
        self.wait(3)

        self.play(
            FadeOut(self.norm_eq),
            FadeOut(self.norm_def),
            self.k_means_eq.animate.set_color_by_tex_to_color_map(
                {"||x_{n} - c_{k}||^2": WHITE}
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
        )
        self.play(
            LaggedStart(
                *[
                    TransformMatchingTex(self.kron_1[1:2].copy(), self.cn_def[0:1]),
                    Write(self.cn_def[1:2]),
                    TransformMatchingTex(self.kron_1[4:5].copy(), self.cn_def[2:]),
                ]
            )
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
        self.label_eq = Tex("\mu_{n}", "= argmin_{k}||{x_{n}} - {c_{k}}||^2")

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
            c_{k} = 
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
        to_isolate = ["A_{nk}", "||x_{n} - c_{k}||^2"]
        self.k_means_eq = Tex(
            "J = \sum_{n=1}^{N} \sum_{k=1}^{K}",
            "A_{nk}",
            "||x_{n} - c_{k}||^2",
            isolate=[*to_isolate],
        )

        self.where = Text("where").scale(0.75)

        # Norm definition
        to_isolate = ["||x_{n} - c_{k}||^2"]
        self.norm_eq = Tex("||x_{n} - c_{k}||^2", isolate=[*to_isolate])
        self.norm_eq.set_color_by_tex_to_color_map({"||x_{n} - c_{k}||^2": BLUE})

        to_isolate = ["${x_{n}}$", "${c_{k}}$"]
        self.norm_def = TexText(
            "Is the squared distance between data point ${x_{n}}$ \\\\"
            "and cluster center ${c_{k}}$",
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
        to_isolate = ["{\mu}", "{x}"]
        self.kron_1 = Tex(
            "1 \\text{ if }",
            "{\mu}_{n}",
            "= k \\text{ (i.e., }",
            "{x}_{n}",
            "\\text{ is a member of cluster } c_{k} \\text{)}",
            isolate=[*to_isolate],
        )
        self.kron_2 = Tex("0 \\text{ otherwise }")

        self.kron_group.add(self.kron_1, self.kron_2)
        self.kron_group.arrange_in_grid(2, 1, 0.1, aligned_edge=LEFT)

        self.A_def_group.add(self.A_def, self.kron_group)
        self.A_def_group.arrange(RIGHT, buff=0.1)

        to_isolate = ["{\mu}", "{x}"]
        self.cn_def = Tex(
            "{\mu}",
            "\\text{ is a vector containing the cluster membership of all }",
            "{x}",
            isolate=[*to_isolate],
        )
        self.cn_def.set_color_by_tex_to_color_map({"{\mu}": TEAL, "{x}": ORANGE}),

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
