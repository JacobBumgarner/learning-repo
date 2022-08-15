from manimlib import *

# from manim import *

# print(manimpango.list_fonts())
points = np.array(
    [
        1.108084720644417,
        -1.2509540934446919,
        -1.1984940501222023,
        1.6103470119956345,
        -0.8491122830124362,
        -1.1935884748756593,
        -0.24662421169449525,
        2.4113311493122764,
        -0.3571086625966007,
        -0.24731511684480006,
        0.7299495445548285,
        -1.2961346535902194,
        1.125731073678829,
        -1.7757990839188238,
        -1.9270827135180693,
        -1.8761494289764047,
        1.182102502649793,
        1.8912032980085758,
        2.1531050064897927,
        -1.0239598465195365,
        1.7574043084852462,
        1.2563597694117072,
        -0.14445079068379432,
        -1.1397397915902852,
        -0.9163821350344091,
        -1.3694348087107537,
        -0.7114665795074704,
        2.014021232704346,
        -0.9042761760220348,
        1.2754559284000837,
        1.274368988807961,
        2.2622679063924394,
        0.9322808102860738,
        0.022825163455644365,
        1.0319237763151985,
        -0.5425520895617242,
        -1.2487797436216759,
        -1.6468422470438135,
        -1.1059279327709703,
        -0.33898032810574996,
        -0.9317202299901362,
        -0.7531140963490046,
        -0.8366519948519304,
        -1.3529531953760048,
        1.6663291077430455,
        -0.19378864354478523,
        -1.5828780053558478,
        1.7086696378439166,
        1.109229688614288,
        0.5866228612068721,
        1.3246996167964051,
        -0.8812428666808865,
        -1.4938379877535422,
        0.22022597440898942,
        3.730138896698768,
        1.1529085204765865,
        -1.8030601479574873,
        2.1932431246719553,
        2.2562981362402126,
        1.2841466238394281,
        -1.3819181875659137,
        0.9074174106009874,
        0.09494982785992426,
        -0.44752785135522033,
        0.9450610444269605,
        -0.3409412054191009,
        -0.6379857403375633,
        -0.3145211938387711,
        -1.4215760496194543,
        -1.4231388572846353,
        -1.0727723601212642,
        0.3949189320107318,
        -1.0759774555653825,
        1.5285913143744703,
        -1.101589161210171,
        3.0976974052420028,
    ]
)


class Linear(Scene):
    def construct(self):
        self.create_left_column()
        self.create_top_right_column()
        self.write_numbers()
        self.group_objects()
        self.animate()
        return

    def animate(self):
        # Left Column
        self.play(Write(self.top_text))
        self.play(FadeIn(self.equation))
        self.wait(0.5)
        self.play(TransformMatchingTex(self.equation.copy(), self.linear))
        self.wait(0.5)
        self.play(TransformMatchingTex(self.linear.copy(), self.definitions))
        self.wait(1)
        self.play(Write(self.line))
        
        # Right Column
        self.play(TransformMatchingTex(self.linear.copy(), self.expanded_equation))

        self.play(Write(self.number_line, run_time=2.5))
        self.animate_points()

        self.wait(1)
        self.play(*[FadeOut(item) for item in self.mobjects])
        return

    def animate_points(self):
        point_numbers = [0, 1, np.arange(2, points.shape[0] - 2).tolist(), -1]

        beneath_point = (
            Dot(self.expanded_equation.get_center()).shift(DOWN).get_center()
        )

        for i in range(len(self.data_line)):
            target_position = self.number_line.number_to_point(points[point_numbers[i]])
            if target_position.ndim == 1:
                target_position = np.expand_dims(target_position, 0)

            self.play(FadeIn(self.data_line[i]))

            # Animate the x data into the equation
            self.play(
                LaggedStart(
                    *[
                        FadeOutToPoint(
                            self.data_line[i].copy(),
                            self.expanded_equation[1:].get_center(),
                        )
                        for _ in range(max(1, int(target_position.shape[0] / 4)))
                    ],
                    run_time=1 if i != 2 else 2,
                    rate_func=rush_from
                )
            )

            # Create the resulting dots
            dots = [
                Dot(
                    self.expanded_equation[0].get_center(),
                    radius=0.05,
                    color=GREEN,
                    opacity=0.75,
                )
                for _ in range(target_position.shape[0])
            ]

            self.play(
                FadeOutToPoint(
                    self.expanded_equation[2:].copy(),
                    self.expanded_equation[0].get_center(),
                ),
                *[FadeIn(dot) for dot in dots]
            )

            # Animate the dots moving onto the numberline
            self.play(
                LaggedStart(
                    *[
                        dot.animate.move_to(target_position[i])
                        for i, dot in enumerate(dots)
                    ],
                    run_time=1 if i != 2 else 3,
                    rate_func=rush_from
                )
            )

    def write_numbers(self):
        isolate = ["x"]
        self.data_line = Tex(
            "x_{\\text{age}} & = 59 & x_{\\text{max HR}} & = 161 & \dots\ &&  x_{\\text{cholesterol}} & = 234 \\\\",
            "x_{\\text{age}} & = 55 & x_{\\text{max HR}} & = 132 & \dots\ &&  x_{\\text{cholesterol}} & = 353 \\\\",
            """& \dots\ & \dots\ && \dots\ && \dots\ \\\\ """,
            "x_{\\text{age}} & = 42 & x_{\\text{max HR}} & = 125 & \dots\ &&  x_{\\text{cholesterol}} & = 315 \\\\",
        ).scale(0.5)

        # 0.25
        # 0.78
        # 0.66

        self.right_group.add(self.data_line, self.expanded_equation, self.number_line)
        self.data_line.to_edge(UP)
        self.data_line.shift(UP)

    def group_objects(self):
        self.group = VGroup()
        self.group.add(self.left_group, self.line, self.right_group)
        self.group.arrange()
        self.group.to_edge(LEFT)

    def create_top_right_column(self):

        to_isolate = [
            "z",
            "w_{1}",
            "w_{2}",
            "w_{n}",
            "x_{1}",
            "x_{2}",
            "x_{n}",
            "b",
        ]
        self.expanded_equation = Tex(
            "z = w_{1} x_{1} + w_{2} x_{2} + \dots\ + w_{n} x_{n} + b",
            isolate=[*to_isolate],
        )
        self.expanded_equation.set_color_by_tex_to_color_map(
            {
                "z": GREEN,
                "w_{1}": BLUE,
                "w_{2}": BLUE,
                "w_{n}": BLUE,
                "x_{1}": RED,
                "x_{2}": RED,
                "x_{n}": RED,
                "b": ORANGE,
            }
        )

        self.number_line = NumberLine(
            color=WHITE, x_range=[-2, 4, 1], include_numbers=True, width=8
        )

        self.right_group = VGroup()
        self.right_group.add(self.expanded_equation, self.number_line)
        self.right_group.arrange(DOWN, buff=1)
        self.expanded_equation.shift(UP * 1.5)
        self.number_line.shift(UP)

    def create_left_column(self):
        self.top_text = Text(
            "Logistic Function: \n" "Linear Component",
        )

        to_isolate = ["z", "W", "x", "b"]
        self.equation = Tex(r"p(y | x) = \sigma(W \cdot x + b)", isolate=[*to_isolate])

        self.linear = Tex(r"z = W \cdot x + b", isolate=[*to_isolate])
        self.linear.set_color_by_tex_to_color_map({"W": BLUE, "b": RED, "x": ORANGE})

        self.definitions = Tex(
            "W",
            "&\\text{: Weights vector}",
            "\\\\",
            "b",
            "&\\text{: Bias scalar}",
            "\\\\",
            "x",
            "&\\text{: Input data}",
            isolate=[
                "W",
                "\\text{: Weights vector}",
                "b",
                "\\text{: Bias scalar}",
                "x",
                "\\text{: Input data}",
            ],
        )
        self.definitions.set_color_by_tex_to_color_map(
            {
                "W": BLUE,
                "b": ORANGE,
                "x": RED,
                "\\text{: Weights vector}": WHITE,
                "\\text{: Bias scalar}": WHITE,
                "\\text{: Input data}": WHITE,
            }
        )

        self.left_group = VGroup()
        self.left_group.add(self.top_text, self.equation, self.linear, self.definitions)
        self.left_group.arrange(DOWN, buff=1)
        self.left_group.to_edge(LEFT)
        self.left_group.to_edge(UP)
        self.left_group.scale(0.9)

        line_top = self.left_group.get_top()
        line_top[0] -= self.left_group.get_right()[0]
        line_bottom = self.left_group.get_bottom()
        line_bottom[0] -= self.left_group.get_right()[0]
        self.line = Line(start=line_top, end=line_bottom)

        return


class LogisticFunction(Scene):
    def construct(self):
        self.create_objects()
        self.animate()

    def animate(self):
        ### INTRO
        self.create_group(self.title, self.equation, self.intro)

        self.play(Write(self.title))
        self.wait(0.5)

        self.play(FadeIn(self.equation))
        self.play(
            self.equation.animate.set_color_by_tex_to_color_map(
                {"y": BLUE, "x": ORANGE}
            )
        )
        self.wait(0.75)
        self.play(FadeIn(self.intro))
        self.wait(1.5)

        ### WEIGHTS
        self.play(FadeOut(self.intro))

        self.play(
            self.equation.animate.set_color_by_tex_to_color_map(
                {"y": WHITE, "W": GREEN, "b": BLUE_C}
            )
        )

        self.create_group(self.title, self.equation, self.weights_and_bias)
        self.play(FadeIn(self.weights_and_bias))
        self.wait(3)

        ### SIGMOID
        self.play(FadeOut(self.weights_and_bias))

        self.play(
            self.equation.animate.set_color_by_tex_to_color_map(
                {"y": BLUE, "W": WHITE, "b": WHITE, "x": WHITE, "\sigma": LIGHT_PINK}
            )
        )

        self.create_group(self.title, self.equation, self.sigmoid)
        self.play(FadeIn(self.sigmoid))
        self.wait(3)

        ### Exit
        self.play(FadeOut(self.sigmoid))
        self.play(FadeOut(self.title))
        self.play(ShrinkToCenter(self.equation))

    def create_objects(self):
        ######
        # INTRO
        #####
        self.title = Title("The Logistic Function", font_size=55)
        # text.set_color_by_gradient(BLUE, ORANGE)

        to_isolate = ["y", "x", "W", "b", "\sigma"]
        self.equation = Tex("p(y | x) = \sigma (W \cdot x + b)", isolate=[*to_isolate])
        self.equation.set_color_by_tex_to_color_map({"y": WHITE, "x": WHITE})

        self.intro = TexText(
            """
            The logistic function produces probabilistic labels ${y}$ \\\\
            for input data ${x}$.
            """,
            isolate=["${x}$", "${y}$"],
        )
        self.intro.set_color_by_tex_to_color_map({"${y}$": BLUE, "${x}$": ORANGE})

        ######
        # Weights and Bias
        #####
        self.weights_and_bias = TexText(
            """
            The function first linearly transforms \\\\
            the input data ${x}$ with the model's learned \\\\
            Weights (${W}$) and bias (${b}$) parameters.
            """,
            isolate=["${x}$", "${W}$", "${b}$", "Weights", "bias"],
        )
        self.weights_and_bias.set_color_by_tex_to_color_map(
            {
                "${x}$": ORANGE,
                "${W}$": GREEN,
                "Weights": GREEN,
                "${b}$": BLUE_C,
                "bias": BLUE_C,
            }
        )

        ######
        # Sigmoid
        #####
        self.sigmoid = TexText(
            """
            The function then applies the non-linear \\\\
            sigmoid (${\sigma}$) transformation to the linear result \\\\
            to produce the probability labels ${y}$. 
            """,
            isolate=["sigmoid", "${\sigma}$", "${y}$"],
        )
        self.sigmoid.set_color_by_tex_to_color_map(
            {"sigmoid": LIGHT_PINK, "${\sigma}$": LIGHT_PINK, "${y}": BLUE}
        )

    def create_group(self, *args):
        vg = VGroup()
        vg.add(*args)
        vg.arrange(DOWN, buff=1)
        vg.to_edge(UP)
        return
