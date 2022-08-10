from manimlib import *
import manimpango

# print(manimpango.list_fonts())

class Linear(Scene):
    def construct(self):
        self.create_objects()
        self.animate()
        return
    
    def animate(self):
        self.add(self.top_text, self.equation)
        return
    
    def create_objects(self):
        self.top_text = Text(
            "Logistic Function: \n"
            "Linear Component",
            # font="Computer MOdern"
        )
        
        self.equation = Tex(
            r"p(y | x) = \sigma(\underbrace{W \cdot x + b}_\text{Linear})"
        )
        
        self.linear = Tex(
            r"Z = W \cdot x + b"
        )
        
        group = VGroup()
        group.add(self.top_text, self.equation)
        group.arrange(DOWN)
        group.to_edge(UP)
        group.to_edge(LEFT)
        
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
        self.play(self.equation.animate.set_color_by_tex_to_color_map(
            {"y": BLUE, "x": ORANGE}
        ))        
        self.wait(0.75)
        self.play(FadeIn(self.intro))
        self.wait(1.5)
        
        ### WEIGHTS
        self.play(FadeOut(self.intro))    
        
        self.play(self.equation.animate.set_color_by_tex_to_color_map(
            {"y": WHITE, "W": GREEN, "b": BLUE_C}
        ))
        
        self.create_group(self.title, self.equation, self.weights_and_bias)
        self.play(FadeIn(self.weights_and_bias))
        self.wait(3)
        
        ### SIGMOID
        self.play(FadeOut(self.weights_and_bias))
        
        self.play(self.equation.animate.set_color_by_tex_to_color_map(
            {"y": BLUE, "W": WHITE, "b": WHITE, "x": WHITE, "\sigma": LIGHT_PINK}
        ))
        
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
        self.equation = Tex(
            "p(y | x) = \sigma (W \cdot x + b)", isolate=[*to_isolate]
        )
        self.equation.set_color_by_tex_to_color_map(
            {"y": WHITE, "x": WHITE}
        )
        
        self.intro = TexText(
            """
            The logistic function produces probabilistic labels ${y}$ \\\\
            for input data ${x}$.
            """,
            isolate=["${x}$", "${y}$"],
        )
        self.intro.set_color_by_tex_to_color_map(
            {"${y}$": BLUE, "${x}$": ORANGE}
        )
        
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
                "${x}$":ORANGE, 
                 "${W}$": GREEN, "Weights": GREEN,
                 "${b}$": BLUE_C, "bias": BLUE_C
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
            isolate=["sigmoid", "${\sigma}$", "${y}$"]
        )
        self.sigmoid.set_color_by_tex_to_color_map(
            {
                "sigmoid": LIGHT_PINK, "${\sigma}$": LIGHT_PINK,
                "${y}": BLUE
            }
        )
        
        
    def create_group(self, *args):
        vg = VGroup()
        vg.add(*args)
        vg.arrange(DOWN, buff=1)
        vg.to_edge(UP)
        return
