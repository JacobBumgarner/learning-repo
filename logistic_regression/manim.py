from manimlib import *

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

sorted_points = np.sort(points)


class BackProp(Scene):
    def construct(self):
        self.construct_first_scene()
        self.construct_second_scene()
        self.construct_dW_scene()
        self.construct_simplification_scene()
        self.construct_dB_scene()
        self.animate()

    def animate(self):
        # First scene
        self.play(Write(self.title))
        self.play(Write(self.intro1, run_time=4))
        self.play(Write(self.intro2))
        self.wait(0.75)
        self.play(FadeOut(self.intro_group))

        # Second scene
        self.play(Write(self.chain_rule_text_W))
        self.wait(0.5)
        self.play(
            TransformMatchingTex(
                self.chain_rule_text_W.copy(), self.chain_rule_equation_W
            )
        )
        self.wait(0.75)

        # Third Scene
        self.play(
            FadeOut(self.chain_rule_text_W),
            # self.chain_rule_equation_W.animate.shift(UP*2),
            self.chain_rule_equation_W.animate.center().to_edge(LEFT),
        )
        self.play(
            self.chain_rule_equation_W.animate.set_color_by_tex_to_color_map(
                {"{dC}": WHITE, "{dW}": WHITE}
            )
        )

        # Third scene - dW
        # self.play(Indicate(self.chain_rule_equation_W[:3]))
        self.play(
            TransformMatchingTex(self.chain_rule_equation_W.copy()[4:5], self.dCdA),
            self.chain_rule_equation_W.animate.set_color_by_tex_to_color_map(
                {
                    "{{dC} \\over {dA}}": PURPLE,
                }
            ),
        )
        self.wait(0.5)

        self.play(
            TransformMatchingTex(
                self.chain_rule_equation_W.copy()[5:6], self.dAdZ1[0:2]
            ),
            self.chain_rule_equation_W.animate.set_color_by_tex_to_color_map(
                {
                    "{{dA} \\over {dZ}}": PINK,
                }
            ),
        )
        self.play(FadeIn(self.dAdZ1[2:]))
        self.play(TransformMatchingTex(self.dAdZ1, self.dAdZ2))
        self.wait(0.5)

        self.play(
            TransformMatchingTex(self.chain_rule_equation_W.copy()[6:], self.dZdW),
            self.chain_rule_equation_W.animate.set_color_by_tex_to_color_map(
                {
                    "{{dZ} \\over {dW}}": TEAL,
                }
            ),
        )
        self.wait(1.5)

        # move them to prep for the next scene
        self.play(
            TransformMatchingTex(
                self.chain_rule_equation_W, self.chain_rule_equation_W_simplification
            ),
            self.dCdA.animate.center().to_edge(DL).shift(UP),
            self.dAdZ2.animate.center().to_edge(DOWN).shift(UP),
            self.dZdW.animate.center().to_edge(DR).shift(UP).shift(LEFT),
        )
        self.wait(0.5)

        # Fourth scene - simplification
        self.play(FadeIn(self.expanded_dCdW[0:3]))

        self.play(
            Indicate(self.chain_rule_equation_W_simplification[3:5]),
            TransformMatchingTex(self.dCdA, self.expanded_dCdW[3:8]),
        )
        self.wait(0.5)
        self.play(
            Indicate(self.chain_rule_equation_W_simplification[5:6]),
            TransformMatchingTex(self.dAdZ2, self.expanded_dCdW[8:9]),
        )
        self.wait(0.5)
        self.play(
            Indicate(self.chain_rule_equation_W_simplification[6:]),
            TransformMatchingTex(self.dZdW, self.expanded_dCdW[9:]),
        )
        self.wait(1)

        self.play(
            TransformMatchingTex(self.expanded_dCdW.copy(), self.simplified_dCdW1)
        )  # could make this prettier with a loop but whatever, just crunching
        self.wait(0.5)
        self.play(
            TransformMatchingTex(self.simplified_dCdW1.copy(), self.simplified_dCdW2)
        )
        self.wait(0.5)
        self.play(
            TransformMatchingTex(self.simplified_dCdW2.copy(), self.simplified_dCdW3)
        )
        self.wait(0.5)
        self.play(
            TransformMatchingTex(
                self.simplified_dCdW3.copy(), self.final_simplified_dCdW
            )
        )
        self.wait(1)

        # clear all, move to center
        self.play(
            FadeOut(self.simplification_group[1:-1]),
            self.final_simplified_dCdW.animate.center().scale(1.25),  # scale back to 1
        )
        self.wait(1)

        # Fifth scene, brief dB
        self.play(FadeOut(self.final_simplified_dCdW))
        self.play(Write(self.chain_rule_text_B))
        self.wait(0.5)
        self.play(
            TransformMatchingTex(
                self.chain_rule_text_B.copy(), self.chain_rule_equation_B
            )
        )
        self.play(FadeIn(self.dZdB))
        self.play(Write(self.tf))
        self.play(FadeIn(self.simplified_dCdB))
        self.wait(1)

        self.play(
            FadeOut(self.dB_group[1:-1]),
            self.simplified_dCdB.animate.center(),
        )
        self.wait(1)

        # Final
        self.play(
            FadeIn(self.final_simplified_dCdW.center().shift(UP)),
            self.simplified_dCdB.animate.center().shift(DOWN),
        )

    def construct_dB_scene(self):
        self.dB_group = VGroup()
        to_isolate = ["{dC}", "{dB}"]
        self.chain_rule_text_B = Tex(
            "\\text{The deriative of C } ( {dC} ) \\text{ w.r.t the bias } ( {dB} )",
            isolate=[*to_isolate],
        )
        self.chain_rule_text_B.set_color_by_tex_to_color_map({"dC": RED, "dB": ORANGE})

        to_isolate = [
            "{dC}",
            "{dB}",
            "{dA}",
            "{dZ}",
            "{{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dB}}",
        ]
        self.chain_rule_equation_B = Tex(
            "{{dC} \\over {dB}}",
            "= {{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dB}}",
            isolate=[*to_isolate],
        )
        self.chain_rule_equation_B.set_color_by_tex_to_color_map(
            {
                "{dC}": RED,
                "{dB}": ORANGE,
                "{{dC} \\over {dA}}": WHITE,
                "{{dA} \\over {dZ}}": WHITE,
                "{{dZ} \\over {dB}}": WHITE,
            }
        )

        self.dZdB = Tex("{{dZ} \\over {dB}}", " = 1", isolate=["{{dZ} \\over {dB}}"])

        self.tf = Tex("\\therefore")

        self.simplified_dCdB = Tex(
            "{{dC} \\over {dB}}", " = (A - Y)", isolate=["{dC}", "{dB}"]
        )
        self.simplified_dCdB.set_color_by_tex_to_color_map(
            {
                "{dC}": RED,
                "{dB}": ORANGE,
            }
        )

        self.dB_group.add(
            self.title,
            self.chain_rule_text_B,
            self.chain_rule_equation_B,
            self.dZdB,
            self.tf,
            self.simplified_dCdB,
        )
        self.dB_group.arrange(DOWN, buff=0.4)
        self.dB_group.to_edge(UP)

    def construct_simplification_scene(self):
        scalar = 0.8
        self.simplification_group = VGroup()

        to_isolate = [
            "{dC}",
            "{dW}",
            "{dA}",
            "{dZ}",
            "{{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dW}}",
        ]
        self.chain_rule_equation_W_simplification = Tex(
            "{{dC} \\over {dW}}",
            "= {{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dW}}",
            isolate=[*to_isolate],
        ).scale(scalar)
        self.chain_rule_equation_W_simplification.set_color_by_tex_to_color_map(
            {
                "{dC}": RED,
                "{dW}": BLUE,
                "{{dC} \\over {dA}}": WHITE,
                "{{dA} \\over {dZ}}": WHITE,
                "{{dZ} \\over {dW}}": WHITE,
            }
        )

        to_isolate = [
            "{dC}",
            "{dB}",
            "A(1-A)",
            "X",
            " = ",
            "{{Y} \\over {A}}",
            "{{1 - Y} \\over {1 - A}}",
        ]
        self.expanded_dCdW = Tex(
            "{{dC} \\over {dW}}",
            " = ",
            " \\left( - {{Y} \\over {A}} + {{1 - Y} \\over {1 - A}} \\right)",
            "A(1-A)",
            "X",
            isolate=[*to_isolate],
        ).scale(scalar)
        self.expanded_dCdW.set_color_by_tex_to_color_map({"{dC}": RED, "{dW}": BLUE})

        to_isolate = ["A(1-A)", "X", "{{Y} \\over {A}}", "{{1 - Y} \\over {1 - A}}"]
        self.simplified_dCdW1 = Tex(
            " = ",
            " \\left( - A(1-A) {{Y} \\over {A}} +  A(1-A) {{1 - Y} \\over {1 - A}} \\right)",
            "X",
            isolate=[*to_isolate],
        ).scale(scalar)

        to_isolate = ["X", "Y", "A"]
        self.simplified_dCdW2 = Tex(
            " = ",
            " ( - Y(1-A) +  A(1-Y) )",
            "X",
            isolate=[*to_isolate],
        ).scale(scalar)

        to_isolate = ["X", "Y", "A"]
        self.simplified_dCdW3 = Tex(
            " = ",
            " ( - Y + AY + A - AY )",
            "X",
            isolate=[*to_isolate],
        ).scale(scalar)

        to_isolate = ["X", "Y", "A", "{dC}", "{dB}"]
        self.final_simplified_dCdW = Tex(
            "{{dC} \\over {dW}}",
            " = ",
            " ( A - Y )",
            "X",
            isolate=[*to_isolate],
        ).scale(scalar)
        self.final_simplified_dCdW.set_color_by_tex_to_color_map(
            {"{dC}": RED, "{dW}": BLUE}
        )

        self.simplification_group.add(
            self.title,
            self.chain_rule_equation_W_simplification,
            self.expanded_dCdW,
            self.simplified_dCdW1,
            self.simplified_dCdW2,
            self.simplified_dCdW3,
            self.final_simplified_dCdW,
        )
        self.simplification_group.arrange(DOWN, buff=0.4)
        self.simplification_group.to_edge(UP)

    def construct_dW_scene(self):
        self.dW_group1 = VGroup()
        self.dW_group2 = VGroup()

        to_isolate = [
            "{dC}",
            "{dA}",
            "{{dC} \\over {dA}}",
            "{{Y} \\over {A}}",
            "{{1 - Y} \\over {1 - A}}",
        ]
        self.dCdA = Tex(
            "{{dC} \\over {dA}}",
            " = -{{Y} \\over {A}} + {{1 - Y} \\over {1 - A}}",
            isolate=[*to_isolate],
        )
        self.dCdA.set_color_by_tex_to_color_map(
            {
                "{{dC} \\over {dA}}": PURPLE,
            }
        )

        to_isolate = ["{dA}", "{dZ}", "{{dA} \\over {dZ}}", "A(1-A)"]
        self.dAdZ1 = Tex(
            "{{dA} \\over {dZ}} &= \\sigma{(Z)}(1 - \\sigma{(Z)})\\\\",
            " &= A(1-A)",
            isolate=[*to_isolate],
        )
        self.dAdZ1.set_color_by_tex_to_color_map(
            {
                "{{dA} \\over {dZ}}": PINK,
            }
        )

        self.dAdZ2 = Tex(
            "{{dA} \\over {dZ}} = A(1-A)",
            isolate=[*to_isolate],
        )
        self.dAdZ2.set_color_by_tex_to_color_map(
            {
                "{{dA} \\over {dZ}}": PINK,
            }
        )

        to_isolate = ["{dZ}", "{dW}", "{{dZ} \\over {dW}}", "X"]
        self.dZdW = Tex("{{dZ} \\over {dW}} = X", isolate=[*to_isolate])
        self.dZdW.set_color_by_tex_to_color_map(
            {
                "{{dZ} \\over {dW}}": TEAL,
            }
        )

        self.dW_group1.add(self.title, self.dCdA, self.dAdZ1)
        self.dW_group1.arrange(DOWN, buff=1)
        self.dW_group1.to_edge(UP)
        self.dW_group2.add(self.title, self.dCdA, self.dAdZ2, self.dZdW)
        self.dW_group2.arrange(DOWN, buff=1)
        self.dW_group2.to_edge(UP)

    def construct_second_scene(self):
        self.backprop_group = VGroup()

        to_isolate = ["{dC}", "{dW}"]
        self.chain_rule_text_W = Tex(
            "\\text{Derivative of C } ( {dC} ) \\text{ w.r.t Weights } ( {dW} )",
            isolate=[*to_isolate],
        )
        self.chain_rule_text_W.set_color_by_tex_to_color_map(
            {"{dC}": RED, "{dW}": BLUE}
        )
        to_isolate = [
            "{dC}",
            "{dW}",
            "{dA}",
            "{dZ}",
            "{{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dW}}",
        ]
        self.chain_rule_equation_W = Tex(
            "{{dC} \\over {dW}}",
            "= {{dC} \\over {dA}}",
            "{{dA} \\over {dZ}}",
            "{{dZ} \\over {dW}}",
            isolate=[*to_isolate],
        )
        self.chain_rule_equation_W.set_color_by_tex_to_color_map(
            {
                "{dC}": RED,
                "{dW}": BLUE,
                "{{dC} \\over {dA}}": WHITE,
                "{{dA} \\over {dZ}}": WHITE,
                "{{dZ} \\over {dW}}": WHITE,
            }
        )

        self.backprop_group.add(
            self.title,
            self.chain_rule_text_W,
            self.chain_rule_equation_W,
        )
        self.backprop_group.arrange(DOWN, buff=1)
        self.backprop_group.to_edge(UP)

    def construct_first_scene(self):
        self.intro_group = VGroup()

        self.title = Title("Logistic Regression Backpropagation", font_size=42)
        self.title.to_edge(UP)

        self.intro1 = MarkupText(
            "The goal of backpropagation is to find the partial derivative of the Cost\n"
            "function with respect to the model's Weights and bias parameters. ",
            alignment="CENTER",
            # isolate=["Weights", "bias"],
            t2c={"Cost": RED, "Weights": BLUE, "bias": ORANGE},
            justify=True,
        ).scale(0.8)

        self.intro2 = MarkupText("This can be achieved with the chain rule.").scale(0.8)
        self.intro_group.add(self.intro1, self.intro2)
        self.intro_group.arrange(DOWN, buff=0.4)


class Sigmoid(Scene):
    def construct(self):
        # Construct the main scene components
        self.create_left_column()
        self.create_right_column()
        self.group_page()

        # Animate the scene
        self.animate()
        return

    def animate(self):
        # Animate the left column
        self.play(Write(self.top_text))
        self.play(FadeIn(self.equation[0:11]))
        self.wait(0.5)
        self.play(
            TransformMatchingTex(self.equation[0:11].copy(), self.equation[11:18])
        )
        self.wait(0.5)
        self.play(TransformMatchingTex(self.equation[11:18].copy(), self.equation[18:]))
        self.wait(1)
        self.play(TransformMatchingTex(self.equation[18:].copy(), self.definitions))

        self.wait(1.5)
        self.play(Write(self.line))

        # Animate the right column
        self.play(Write(self.graph), Write(self.number_line_group))

        self.animate_dots_on_line()
        self.wait(0.5)
        self.animate_sigmoid_function()
        self.animate_dots_to_sigmoid()

        self.play(
            FadeOut(self.number_line_group, DOWN),
            self.graph.animate.shift(DOWN * 0.8),
            self.sigmoid_graph.animate.shift(DOWN * 0.8),
            *[dot.animate.shift(DOWN * 0.8) for dot in self.dots]
        )

        # Move the sigmoid-transformed data into their own groups
        # self.animate_label_indicators()

        self.animate_dots_to_groups()
        self.wait(0.5)
        # Fade the graph, add the "Disease"
        self.play(
            FadeOut(self.graph),
            FadeOut(self.sigmoid_graph),
            *[dot.animate.shift(DOWN * 3) for dot in self.dots],
            self.positive_label_group.animate.shift(UP * 1.5),
            self.negative_label_group.animate.shift(UP * 1.5)
        )

        self.animate_prediction_text()

        self.wait(1)
        self.play(*[FadeOut(item) for item in self.mobjects])
        return

    def animate_prediction_text(self):
        self.prediction_title = Text("Model Predictions:", font_size=42)
        self.prediction_title.move_to(self.graph.get_center()).shift(UP * 1.8)

        self.positive_prediction = Text("Heart Disease", font_size=32)
        self.positive_prediction.move_to(self.positive_arrow.get_center()).shift(UP * 2)

        self.negative_prediction = Text("No Heart Disease", font_size=32)
        self.negative_prediction.move_to(self.negative_arrow.get_center()).shift(UP * 2)

        self.play(Write(self.prediction_title))
        self.play(Write(self.negative_prediction))
        self.play(Write(self.positive_prediction))

    def animate_dots_to_groups(self):
        x = np.linspace(-3.5, -0.5, 10)
        y = np.linspace(1.3, 1.1, 5)

        x, y = np.meshgrid(x, y)
        neg_coords = np.stack((x.flatten(), y.flatten()), axis=1)[:42]  # 42 neg points

        x = np.linspace(0.5, 3.5, 10)
        y = np.linspace(1.3, 1.1, 5)

        x, y = np.meshgrid(x, y)
        pos_coords = np.stack((x.flatten(), y.flatten()), axis=1)[:34]  # 34 pos points

        sorted_coords = np.append(neg_coords, pos_coords, axis=0)

        # Left Label
        self.negative_label_group = VGroup()
        left_arrow_start = self.graph.get_bottom()
        left_arrow_start[0] = self.graph.coords_to_point(0.2, 0)[0]
        left_arrow_end = np.array(
            [self.graph.get_left()[0], self.graph.get_bottom()[1]]
        )

        self.negative_arrow = Arrow(left_arrow_start, left_arrow_end, color=RED)
        self.negative_arrow.shift(DOWN * 0.3)

        self.negative_text = Text("Negative Labels", font_size=28)
        self.negative_text.move_to(self.negative_arrow.get_center()).shift(DOWN * 0.5)

        self.negative_eq = Tex("\hat{y} < 0.5", isolate=["\hat{y}"]).scale(0.75)
        self.negative_eq.set_color_by_tex_to_color_map({"\hat{y}": BLUE})
        self.negative_eq.move_to(self.negative_text.get_center()).shift(DOWN * 0.5)

        self.negative_label_group.add(
            self.negative_arrow, self.negative_text, self.negative_eq
        )
        self.play(Write(self.negative_label_group))

        self.play(
            LaggedStart(
                *[
                    dot.animate.move_to(
                        self.graph.coords_to_point(*sorted_coords[i])
                    ).set_color(RED)
                    for i, dot in enumerate(self.dots[:42])
                ]
            )
        )

        # Right Label
        self.positive_label_group = VGroup()

        right_arrow_start = self.graph.get_bottom()
        right_arrow_start[0] = self.graph.coords_to_point(-0.3, 0)[0]
        right_arrow_end = np.array(
            [self.graph.get_right()[0], self.graph.get_bottom()[1]]
        )
        self.positive_arrow = Arrow(right_arrow_start, right_arrow_end, color=GREEN)
        self.positive_arrow.shift(DOWN * 0.3)

        self.positive_text = Text("Positive Labels", font_size=28)
        self.positive_text.move_to(self.positive_arrow.get_center()).shift(DOWN * 0.5)

        self.positive_eq = Tex("\hat{y} \geq 0.5", isolate=["\hat{y}"]).scale(0.75)
        self.positive_eq.set_color_by_tex_to_color_map({"\hat{y}": BLUE})
        self.positive_eq.move_to(self.positive_text.get_center()).shift(DOWN * 0.5)

        self.positive_label_group.add(
            self.positive_arrow, self.positive_text, self.positive_eq
        )

        self.play(Write(self.positive_label_group))
        self.play(
            LaggedStart(
                *[
                    dot.animate.move_to(
                        self.graph.coords_to_point(*sorted_coords[42 + i])
                    ).set_color(GREEN)
                    for i, dot in enumerate(self.dots[42:])
                ]
            )
        )

    # def animate_label_indicators(self):

    def animate_dots_to_sigmoid(self):
        sigmoid_values = 1 / (1 + np.exp(-sorted_points))

        self.play(
            LaggedStart(
                *[
                    dot.animate.move_to(
                        self.graph.coords_to_point(sorted_points[i], sigmoid_values[i])
                    ).set_color(BLUE)
                    for i, dot in enumerate(self.dots)
                ]
            )
        )

    def animate_dots_on_line(self):
        self.dots = [
            Dot(
                self.number_line.number_to_point(sorted_points[i]),
                radius=0.05,
                color=ORANGE,
            )
            for i in range(sorted_points.shape[0])
        ]
        self.play(LaggedStart(*[FadeIn(dot) for dot in self.dots]), run_time=1.9)

    def animate_sigmoid_function(self):
        text = Text("Sigmoid Function", font_size=42)
        text.move_to(self.graph.get_top()).shift(UP * 0.3)
        self.sigmoid_graph = self.graph.get_graph(
            lambda x: (1 / (1 + np.exp(-x))), color=PURPLE
        )
        self.play(Write(text))
        self.play(ShowCreation(self.sigmoid_graph))
        self.wait(0.5)
        self.play(FadeOut(text))
        return

    def create_right_column(self):
        self.number_line_group = VGroup()
        self.number_line = NumberLine(
            color=WHITE, x_range=[-2, 4, 1], include_numbers=True, width=8
        )
        self.number_line.shift(DOWN)

        self.number_line_label = Text(
            "Output from Linear Transformations", font_size=24
        )
        self.number_line_label.move_to(self.number_line.get_center()).shift(DOWN * 0.6)

        self.number_line_group.add(self.number_line, self.number_line_label)

        self.graph = Axes(
            (-4, 4, 1),
            (0, 1, 0.25),
            height=4,
            width=6,
            axis_config={"stroke_color": WHITE, "stroke_width": 2},
        )
        self.graph.add_coordinate_labels(font_size=16, num_decimal_places=2)

        self.right_group = VGroup()
        self.right_group.add(self.graph, self.number_line_group)
        self.right_group.arrange(DOWN, buff=1)
        return

    def create_left_column(self):
        self.top_text = Text("Logistic Function:\n" "Sigmoid Activation")

        to_isolate = ["\sigma", "W", "x", "b", "z", "\hat{y}"]
        self.equation = Tex(
            "p(y|x) &= \sigma (W \cdot x + b) \\\\"
            "z &= W \cdot x + b \\\\"
            "\sigma ( z ) &= \hat{y} = {1 \over 1 + e^{-z}}",
            isolate=[*to_isolate],
        )

        self.equation.set_color_by_tex_to_color_map(
            {"\hat{y}": BLUE, "z": ORANGE, "\sigma": PURPLE}
        )

        to_isolate = ["\hat{y}", "z"]
        self.definitions = Tex(
            "\sigma",
            "&\\text{: Sigmoid function}",
            "\\\\",
            "z",
            "&\\text{: }\\parbox{3cm}{Output from linear transformation}",
            "\\\\",
            "\hat{y}",
            "&\\text{: Probabilistic label}",
        ).scale(0.9)
        self.definitions.set_color_by_tex_to_color_map(
            {"\hat{y}": BLUE, "z": ORANGE, "\sigma": PURPLE}
        )

        self.left_group = VGroup()
        self.left_group.add(self.top_text, self.equation, self.definitions)
        self.left_group.arrange(DOWN, buff=0.6)
        self.left_group.to_edge(LEFT)
        self.left_group.to_edge(UP)

        line_top = self.left_group.get_top()
        line_top[0] -= self.left_group.get_right()[0]
        line_bottom = self.left_group.get_bottom()
        line_bottom[0] -= self.left_group.get_right()[0]
        self.line = Line(start=line_top, end=line_bottom)
        self.line.shift(RIGHT)

        return

    def group_page(self):
        self.page_group = VGroup()
        self.page_group.add(self.left_group, self.line, self.right_group)
        self.page_group.arrange()
        self.page_group.to_edge(LEFT)
        self.left_group.to_edge(UP)
        return


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
                    fill_opactiy=0.75,
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
        self.data_line = Tex(
            "x_{\\text{age}} & = 59 & x_{\\text{max HR}} & = 161 & \dots\ &&  x_{\\text{cholesterol}} & = 234 \\\\",
            "x_{\\text{age}} & = 55 & x_{\\text{max HR}} & = 132 & \dots\ &&  x_{\\text{cholesterol}} & = 353 \\\\",
            """& \dots\ & \dots\ && \dots\ && \dots\ \\\\ """,
            "x_{\\text{age}} & = 42 & x_{\\text{max HR}} & = 125 & \dots\ &&  x_{\\text{cholesterol}} & = 315 \\\\",
        ).scale(0.5)

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
