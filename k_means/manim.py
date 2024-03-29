# Dance to import our logit model
import os
import sys

module_path = os.path.abspath(os.path.join("."))
sys.path.append(module_path)

from voronoi_processing import get_polygons

# External imports
from manimlib import *
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class KMeansFrameSelection_0(Scene):
    def initialize_parameters_0(self):
        """Initialize the parameters for the animation."""
        self.bee_video = np.load(
            "/Users/jacobbumgarner/Desktop/learning-repo/local_files/k_means/bees_original.npy"
        )
        return

    def construct(self, active_scene=True):
        self.initialize_parameters_0()
        self.construct_title()
        self.construct_video_credit()
        self.construct_text()

        if active_scene:
            self.animate()
        return

    def animate(self):
        self.play(Write(self.title_group))
        self.wait(1)
        self.play(Write(self.video_credit[:]))
        self.wait(0.5)
        self.play(Write(self.text_1[:]))
        self.wait(2)
        self.play(FadeOut(self.text_1), FadeOut(self.video_credit))
        self.play(Write(self.text_2[:]))
        self.wait(2)
        self.play(FadeOut(self.text_2))
        return

    def construct_text(self):
        self.text_1 = (
            TexText(
                """
            Goal: Identify unique keyframes from the video
            for manual feature labeling.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
        )

        self.text_2 = (
            TexText(
                """
            The labeled keyframes will be used to
            train convolutional neural networks
            for automated point tracking.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        return

    def construct_video_credit(self):
        self.video_credit = TexText(
            """
            Video Source:\\\\
            Pereira et al., 2022\\\\
            sleap.ai/datasets.html
            """
        ).scale(0.6).shift(RIGHT*5.3)
        
        return

    def construct_title(self):
        self.title = (
            Text("K-Means for Video Keyframe Extraction").scale(0.8).to_edge(UP)
        )
        self.title.shift(UP * 0.2)

        line_left = LEFT_SIDE + [0.5, 0, 0]
        line_left[1] = self.title.get_bottom()[1] * 0.95
        line_right = RIGHT_SIDE - [0.5, 0, 0]
        line_right[1] = self.title.get_bottom()[1] * 0.95
        self.title_underline = Line(line_left, line_right, color=WHITE, stroke_width=2)

        self.title_group = VGroup(self.title, self.title_underline)
        return


class KMeansFrameSelection_1(KMeansFrameSelection_0):
    def initialize_parameters_1(self):
        self.bee_pca = np.load("label_videos/bee_video_pca.npy")
        self.bee_pca /= 1000
        return

    def load_previous_scene(self):
        # Get the title from scene 0
        self.add(self.title_group)
        return

    def construct(self, active_scene=True):
        """Construct the scene."""
        super().construct(active_scene=False)
        self.initialize_parameters_1()
        if active_scene:
            self.load_previous_scene()

        self.frame_range = [0, 20]
        self.frame_range = [0, self.bee_pca.shape[0]]

        self.construct_PCA_box()
        self.construct_PCA_text()
        self.construct_dots()
        self.construct_graph(pre_animation=active_scene)

        if active_scene:
            self.animate()
        return

    def animate(self):
        """Animate the scene."""
        self.play(LaggedStart(Write(self.PCA_group), Write(self.graph), lag_ratio=0.5))
        self.play(Write(self.PCA_text[:]))
        self.wait(1)
        self.play(Indicate(self.PCA_group, scale_factor=1.2, color=WHITE))
        self.wait(0.25)

        # Animate frame conversion
        self.convert_frames_to_points(self.frame_range)
        self.play(FadeOut(self.PCA_group), FadeOut(self.PCA_text))

        # Center the graph
        self.animate_graph_centering()
        return

    def center_graph(self):
        self.graph_group.move_to(ORIGIN).scale(1.15)
        return

    def animate_graph_centering(self):
        # need to keep the graph group on top... how?
        self.play(self.graph_group.animate.move_to(ORIGIN).scale(1.15))
        return

    def convert_frames_to_points(self, frame_range: list = [0, 300]):
        """Animate the bee video via a construction of"""
        frames = []
        frame_animations = []

        dot_animations = []
        dot_index = 0

        for i in range(*frame_range):
            if i % 40 == 0 or i == (frame_range[1] - 1):
                # if i % 2 == 0:
                image_number = "%04d" % i
                filename = image_number + ".png"
                frame = ImageMobject(filename).scale(0.5).shift(LEFT * 4)
                frames.append(frame)
                animation = frame.animate.move_to(self.PCA_group.get_left()).scale(0)
                frame_animations.append(animation)

            # don't want the values up at 5000 - manim doesn't like large graphs.
            coord = self.bee_pca[i]
            point = self.graph.coords_to_point(*coord)
            dot_animations.append(self.dots[dot_index].animate.move_to(point))
            dot_index += 1

        # Add the frames in reverse order
        self.add(*frames)
        dot_copy = self.PCA_dot.copy()
        dot_animations.append(FadeOut(dot_copy))

        # Animate
        self.play(
            LaggedStart(
                LaggedStart(*frame_animations, lag_ratio=5 / len(frames)),
                LaggedStart(*dot_animations, lag_ratio=5 / len(self.dots)),
                lag_ratio=0.1,
            )
        )

        self.remove(*frames)
        return

    def construct_graph(self, pre_animation=True):
        self.graph = Axes([-5.25, 5.25, 1.5], [-4.25, 4.25, 1.7], width=5, height=4.5)
        self.graph.shift(RIGHT * 4)

        # update dot positions
        if not pre_animation:
            for i, dot in zip(range(*self.frame_range), self.dots):
                coord = self.bee_pca[i]
                position = self.graph.coords_to_point(*coord)
                dot.move_to(position)

        self.graph_group = VGroup()
        self.graph_group.add(self.graph, *self.dots)
        return

    def construct_dots(self):
        total_frames = self.frame_range[1] - self.frame_range[0]

        self.dots = []

        cmap = plt.get_cmap("plasma")
        colors = cmap(np.linspace(0, 1, total_frames))

        for i in range(*self.frame_range):
            position = self.box.get_right()
            dot = Dot(position, color=clr.to_hex(colors[i]), radius=0.03)
            self.dots.append(dot)

        return

    def construct_PCA_text(self):
        self.PCA_text = TexText(
            """
                First, reduce the dimensionality of the video\\\\
                frames using principal component analysis.
            """
        ).scale(0.7)
        self.PCA_text.to_edge(DOWN)
        return

    def construct_PCA_box(self):
        self.box = Rectangle(3.65, 1.75)
        self.PCA_text = TexText("PCA \\\\ ${n_{components} = 2}$")
        self.PCA_group = VGroup(self.box, self.PCA_text).scale(0.6)
        self.PCA_square = Square(0.1, fill_color=WHITE, fill_opacity=1).move_to(
            self.box.get_left()
        )
        self.PCA_dot = Dot(self.box.get_right(), color=WHITE, radius=0.05)
        self.PCA_group.add(self.PCA_dot, self.PCA_square)
        return


class KMeansFrameSelection_2(KMeansFrameSelection_1):
    def initialize_parameters_2(self):
        self.DOT_COLOR = "#a7b8c7"
        return

    def load_previous_scene_1(self):
        self.add(self.title_group)
        self.add(self.graph)
        self.add(*self.dots)
        self.center_graph()
        return

    def construct(self, active_scene=True):
        super().construct(active_scene=False)

        self.initialize_parameters_2()
        self.load_previous_scene_1()

        self.construct_text()

        if active_scene:
            self.animate()
        else:
            self.grayout_dots()
        return

    def animate(self):
        self.play(Write(self.text[:]))
        self.wait(0.5)
        self.animate_dot_grayout()

        self.play(FadeOut(self.text))
        return

    def construct_text(self):
        self.text = (
            TexText(
                """
                Second, cluster the data using k-means to find groups of unique keyframes.\\\\
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        return

    def grayout_dots(self):
        for dot in self.dots:
            dot.set_color(self.DOT_COLOR)
        return

    def animate_dot_grayout(self):
        animations = []
        dot_runtime = 0.2
        for dot in self.dots:
            animations.append(FadeToColor(dot, self.DOT_COLOR, run_time=dot_runtime))

        # lagged_start runtime:
        #   total_runtime = runtime + (#_objects * lag_ratio * runtime)
        # solve for lag_ratio
        total_runtime = 3
        self.play(
            LaggedStart(
                *animations,
                lag_ratio=(total_runtime - dot_runtime)
                / (len(self.dots) * dot_runtime),
            )
        )
        return


class KMeansFrameSelection_3(KMeansFrameSelection_2):
    def initialize_parameters(self):
        self.centroid_history = np.load(
            "label_videos/interpolated_centroid_history.npy"
        )
        self.centroid_history /= 1000

        cmap = plt.get_cmap("jet")
        colors = cmap(np.linspace(0, 1, self.centroid_history.shape[1]))
        self.colors = [clr.to_hex(color) for color in colors]

        cmap = plt.get_cmap("rainbow")
        colors = cmap(np.linspace(0, 1, self.centroid_history.shape[1]))
        colors = np.flip(colors, axis=0)
        self.C_COLORS = [clr.to_hex(colors[i]) for i in range(len(colors))]

        hsv = clr.rgb_to_hsv(colors[..., :3])
        hsv_alterations = np.full(self.centroid_history.shape[1], 0.3)
        hsv_alterations[[0, 4, 7]] += 0.1
        hsv_alterations[[5]] += 0.2
        hsv_alterations[[3]] -= 0.1
        hsv[:, 1] -= hsv_alterations
        dot_colors = clr.hsv_to_rgb(hsv)
        self.D_COLORS = [clr.to_hex(dot_colors[i]) for i in range(len(dot_colors))]
        return

    def get_labels(self, step=0):
        """Return an array with the centroid labels for each dot based on proximity."""
        # dot_positions = np.array([dot.get_center() for dot in self.dots])
        # centroid_positions = np.array([c.get_center() for c in self.centroids])

        distances = cdist(self.bee_pca, self.centroid_history[step])
        labels = np.argmin(distances, axis=1)

        return labels

    # First create centroid persistant animations
    def create_persistent_animations(self, centroids_only=False):
        animations = []  # keeps centroids above lines

        if not centroids_only:
            for dot in self.dots:
                animations.append(dot.animate.set_color(dot.get_color()))
        for centroid in self.centroids:
            animations.append(centroid.animate.set_color(centroid.get_color()))

        return AnimationGroup(*animations)

    def construct(self, active_scene=True):
        super().construct(active_scene=False)

        self.initialize_parameters()

        if active_scene:
            self.construct_text()
            self.construct_algorithm_group()
            self.construct_centroids()
            self.construct_polygons()

            self.animate()
        return

    def animate(self):
        self.play(Write(self.text[:]))
        self.play(Write(self.algo_group[:]))
        self.animate_algorithm_step(0)
        self.animate_centroid_init()
        self.play(FadeOut(self.text))

        for i in range(1, 7):
            # animate label assignment
            self.animate_algorithm_step(1)
            self.wait(0.25)
            if i == 1:
                self.animate_polygons()
            self.animate_label_assignment((i - 1) * 30)
            self.wait(0.25)

            # animate centroid position updates
            self.animate_algorithm_step(2)
            self.wait(0.25)
            self.animate_centroid_position_line_draw(i * 30)
            self.wait(0.25)
            for j in range(30):
                step = j + (i - 1) * 30
                self.animate_centroid_position_update(step)
            self.animate_centroid_position_line_fade()

            self.animate_algorithm_step(3)

        self.play(FadeOut(self.algo_group, LEFT), FadeOut(self.box, LEFT))
        self.play(
            LaggedStart(
                *[Uncreate(polygon) for polygon in self.polygons], lag_ratio=2 / 8
            ),
            self.create_persistent_animations(),
        )
        self.play(*[FadeOut(centroid, scale=0) for centroid in self.centroids])
        return

    def animate_centroid_position_update(self, step):
        self.remove(*self.polygons, *self.centroids)
        self.construct_polygons(step)
        self.construct_centroids(step)
        self.add(*self.polygons, *self.dots, *self.centroids)
        self.wait(1 / 30)
        return

    def animate_centroid_position_line_fade(self):
        animations = []
        for line in self.avg_lines:
            animation = line.animate.put_start_and_end_on(
                line.get_end(), line.get_end()
            )
            animations.append(animation)

        self.play(
            AnimationGroup(*animations),
            self.create_persistent_animations(),
        )
        self.remove(*self.avg_lines)
        return

    def animate_centroid_position_line_draw(self, step):
        """Animate the update to the centroid positions."""
        ### IMPORTANT ###
        # I interpolated one second of frames at 30 fps between the original
        # centroid positions.
        # This means that i * 30 for i in range(original_hist.shape[0]) matches
        # the location of the original positions
        ### IMPORTANT ###
        new_positions = self.centroid_history[step]
        new_positions = [
            self.graph.coords_to_point(*position) for position in new_positions
        ]

        # Aline lines from the dots to the new averaged position
        labels = self.get_labels()
        self.avg_lines = []
        for i, dot in enumerate(self.dots):
            label = labels[i]
            line = Line(
                dot.get_center(),
                new_positions[label],
                color=self.D_COLORS[label],
                stroke_width=0.5,
            )
            self.avg_lines.append(line)

        self.play(
            AnimationGroup(*[Write(line) for line in self.avg_lines]),
            self.create_persistent_animations(),
            # run_time=0.5,
        )
        return

    def animate_label_assignment(self, step):
        labels = self.get_labels(step)

        dot_animations = []
        centroid_animations = []
        for i, centroid in enumerate(self.centroids):
            dot_indices = np.argwhere(labels == i).reshape(-1)
            dots = [self.dots[dot_index] for dot_index in dot_indices]

            for dot in dots:
                dot_animations.append(FadeToColor(dot, color=self.D_COLORS[i]))

            centroid_indicate = Indicate(
                centroid, scale_factor=1.75, color=centroid.get_color()
            )
            centroid_animations.append(centroid_indicate)

        self.play(*centroid_animations)
        self.play(*dot_animations)
        return

    def animate_polygons(self):
        self.play(
            LaggedStart(
                *[Write(polygon) for polygon in self.polygons], lag_ratio=2 / 8
            ),
            self.create_persistent_animations(),
        )
        return

    def animate_centroid_init(self):
        animations = []
        for centroid, dot in zip(self.centroids, self.original_dots):
            animation = LaggedStart(
                Indicate(dot, scale_factor=6, color=centroid.get_color()),
                FadeIn(centroid),
            )
            animations.append(animation)

        self.play(LaggedStart(*animations, lag_ratio=3 / len(self.centroids)))

        return

    def construct_polygons(self, step=0):
        self.polygons = []
        polygons_coords = get_polygons(
            self.centroid_history[step], [-5.25, 5.25], [-4.25, 4.25]
        )
        for i, coords in enumerate(polygons_coords):
            updated_coords = [self.graph.coords_to_point(*point) for point in coords]
            polygon = Polygon(
                *updated_coords, fill_color=self.D_COLORS[i], fill_opacity=0.2
            )
            self.polygons.append(polygon)
        return

    def construct_centroids(self, step=0):
        self.centroids = []
        self.original_dots = []

        for i in range(self.centroid_history.shape[1]):
            coord = self.centroid_history[step, i]
            position = self.graph.coords_to_point(*coord)

            centroid = Dot(position, color=self.C_COLORS[i], radius=0.125)
            self.centroids.append(centroid)

            # Identify the first dot for the animation
            if step == 0:
                original_dot_index = np.argwhere(
                    self.bee_pca[:, :] == self.centroid_history[step, i, :2]
                )[0, 0]
                original_dot = self.dots[original_dot_index]
                self.original_dots.append(original_dot)

        return

    def animate_algorithm_step(self, step):
        if step == 0:
            self.box = None
        new_box = Rectangle(
            self.algo_steps[step].get_width() * 1.05,
            self.algo_steps[step].get_height() * 1.3,
            color=RED,
        )
        new_box.move_to(self.algo_steps[step].get_center())

        if step == 0:
            self.play(Write(new_box))
        else:
            self.play(ReplacementTransform(self.box, new_box))
        self.box = new_box
        return

    def construct_algorithm_group(self):
        self.algo_title_text = TexText("K-means Steps").scale(0.75)
        algo_title_bottom = self.algo_title_text.get_bottom()[1]
        line_left = self.algo_title_text.get_left()
        line_left[1] = algo_title_bottom
        line_right = self.algo_title_text.get_right()
        line_right[1] = algo_title_bottom
        self.algo_title_UL = Line(line_left, line_right)
        self.algo_title_group = VGroup(self.algo_title_text, self.algo_title_UL)

        self.algo_steps = TexText(
            "1. Select centroids\\\\",
            "2. Assign data labels\\\\",
            "3. Update centroid positions\\\\",
            "4. Repeat steps 2 \& 3\\\\until convergence",
        ).scale(0.75)

        self.algo_group = VGroup(self.algo_title_group, self.algo_steps)
        self.algo_group.arrange(DOWN, buff=0.2)
        self.algo_group.scale(0.8).to_edge(LEFT).shift(LEFT * 0.2)
        return

    def construct_text(self):
        self.text = (
            TexText(
                """
                Use $k=8$ to identify 8 keyframes from the video for labeling.\\\\
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        return


class KMeansFrameSelection_4(KMeansFrameSelection_3):
    def initialize_parameters_4(self):
        self.selected_dot_indices = np.load("label_videos/label_frame_indices.npy")
        return

    def load_previous_scene_4(self):
        labels = self.get_labels(180)
        for i, dot in enumerate(self.dots):
            dot.set_color(self.D_COLORS[labels[i]])

        self.add(self.title_group)
        self.add(self.graph)
        self.add(*self.dots)
        return

    def load_frames(self, path):
        images = []
        for frame in self.selected_dot_indices:
            image_name = "%04d" % frame + ".png"
            filename = path + image_name
            image = ImageMobject(filename).scale(0.45)
            images.append(image)
        return images

    def construct(self, active_scene=True):
        super().construct(active_scene=False)
        self.initialize_parameters_4()

        if active_scene:
            self.load_previous_scene_4()
            self.construct_text()
            self.animate()
        return

    def animate(self):
        # select frames
        self.play(Write(self.text_1[:]))
        self.animate_shift_graph_left()
        self.animate_frame_selection()

        # 'label' frames
        self.play(FadeOut(self.text_1))
        self.play(*self.animate_graph_fade(), *self.animate_image_group_centering())
        self.play(Write(self.text_2[:]))
        self.animate_frame_labeling()
        self.play(FadeOut(self.text_2))

        # 'train' cnn
        self.play(Write(self.text_3[:]))
        self.animate_frame_left_stack()
        self.animate_construct_pose_model()
        self.animate_frame_training()
        self.play(FadeOut(self.text_3))

        # 'label' video
        self.play(Write(self.text_4[:]))
        self.wait(1)
        self.play(FadeOut(self.pose_model))

        self.wait(1)
        self.play(FadeOut(self.text_4))
        self.wait(1)
        self.play(FadeOut(self.title_group))

        return
    
    def animate_frame_training(self):
        # animate the movement of the frames into the pca box
        position = self.pose_square_l.get_center()
        
        animations = []
        for i in range(len(self.labeled_images)-1, -1, -1):
            animation = self.labeled_images[i].animate.move_to(position).scale(0)
            animations.append(animation)
        
        self.play(
            LaggedStart(*animations, lag_ratio=2/len(self.original_images)),
            FadeToColor(self.pose_box, GREEN, run_time=(2+(8*(2/8)))),
            FadeToColor(self.pose_square_l, GREEN, run_time=(2+(8*(2/8)))),
            FadeToColor(self.pose_square_r, GREEN, run_time=(2+(8*(2/8)))),
        )
        return

    def animate_construct_pose_model(self):
        self.pose_text = TexText("Pose \\\\ Estimation \\\\ Model").scale(0.7)
        self.pose_box = Rectangle(
            self.pose_text.get_width() * 1.2, self.pose_text.get_height() * 1.2
        )
        self.pose_square_l = Square(0.1, fill_color=WHITE, fill_opacity=1).move_to(
            self.pose_box.get_left()
        )
        self.pose_square_r = Square(0.1, fill_color=WHITE, fill_opacity=1).move_to(
            self.pose_box.get_right()
        )
        self.pose_model = VGroup(
            self.pose_text, self.pose_box, self.pose_square_l, self.pose_square_r
        )
        self.play(Write(self.pose_model[:]))

        return

    def animate_frame_left_stack(self):
        # make the image grid
        scaling = 0.8
        width = self.labeled_images[0].get_width() * 1.1 * scaling / 2
        height = self.labeled_images[0].get_height() * scaling / 1.2
        x = [-width, width]
        y = np.linspace(-height * 2, height * 2, 4)
        image_coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        image_coords = np.pad(image_coords, ((0, 0), (0, 1)))
        image_coords += ORIGIN + DOWN * 0.1 + LEFT * 4

        # animate the image movement & scaling
        animations = []
        for i in [2, 1, 0, 5, 4, 3, 6, 7]:
            animation = (
                self.labeled_images[i].animate.scale(scaling).move_to(image_coords[i])
            )
            animations.append(animation)

        self.play(LaggedStart(*animations))

        return

    def animate_frame_labeling(self):
        # load and position the labeled frames
        path = "/Users/jacobbumgarner/Desktop/learning-repo/local_files/k_means/labeled_bee_images/"
        self.labeled_images = self.load_frames(path)

        image_grid = self.load_image_grid(self.labeled_images)
        for i, image in enumerate(self.labeled_images):
            image.move_to(image_grid[i])

        # animate the 'labeling' of the first frame
        middle_image = 4
        self.add(self.original_images[middle_image])
        self.play(self.original_images[middle_image].animate.scale(3))
        self.labeled_images[middle_image].scale(3)
        self.play(FadeIn(self.labeled_images[middle_image]))
        self.wait(1)
        self.remove(self.original_images[middle_image])
        self.play(self.labeled_images[middle_image].animate.scale(1 / 3))

        # animate the 'labeling' of the rest of the frames. try a fadein first
        # pull all of the labeled images except for the 3rd
        images = [self.labeled_images[i] for i in [2, 1, 0, 5, 3, 6, 7]]
        self.play(*[FadeIn(image, run_time=0.1) for image in images])
        self.play(*[WiggleOutThenIn(image) for image in images])
        self.remove(*self.original_images)
        return

    def animate_image_group_centering(self):
        animations = []
        image_grid = self.load_image_grid(self.original_images)
        for i, image in enumerate(self.original_images):
            animations.append(image.animate.move_to(image_grid[i]))

        return animations

    def animate_graph_fade(self):
        animations = []
        animations.append(FadeOut(self.graph, LEFT))
        animations.extend([FadeOut(dot, LEFT) for dot in self.dots])
        return animations

    def load_image_grid(self, images, right_shift=False):
        # Construct the grid (3, 3, 2)
        height = images[0].get_height() * 1.1
        width = images[0].get_width() * 1.1
        a = [-width, 0, width]
        b = [height, 0]
        top_two_rows = np.array(np.meshgrid(a, b)).T.reshape(-1, 2)
        sorted_indices = np.argsort(top_two_rows[:, 1])
        top_two_rows = np.flip(top_two_rows[sorted_indices], axis=0)
        bottom_rows = [[-width / 2, -height], [width / 2, -height]]
        image_coords = np.concatenate(
            (top_two_rows, bottom_rows),
        )
        image_coords = np.pad(image_coords, ((0, 0), (0, 1)))
        image_coords += ORIGIN + UP * 0.1

        if right_shift:
            image_coords += RIGHT * 3.5

        return image_coords

    def animate_frame_selection(self):
        # First load the images and create their final image grid
        path = "/Users/jacobbumgarner/Desktop/learning-repo/local_files/k_means/bee_images/"
        self.original_images = self.load_frames(path)

        image_grid = self.load_image_grid(self.original_images, right_shift=True)

        # then find the position of the dots associated with each image, set each
        # position as the starting point for each image, make sure to scale the image
        selected_dots = []
        for i, index in enumerate(self.selected_dot_indices):
            selected_dots.append(self.dots[index])
            dot_position = self.dots[index].get_center()
            self.original_images[i].move_to(dot_position).scale(0.45 / 100)

        # lastly, animate the 'selection' of the frames
        animations = []
        for i in range(len(self.original_images)):
            dot_animation = Indicate(selected_dots[i], color=YELLOW, scale_factor=8)
            frame_animation = (
                self.original_images[i].animate.move_to(image_grid[i]).scale(100 / 0.45)
            )
            animation = LaggedStart(dot_animation, frame_animation, lag_ratio=0.6)
            animations.append(animation)

        self.play(LaggedStart(*animations, lag_ratio=2 / len(self.original_images)))
        return

    def animate_shift_graph_left(self):
        self.play(
            self.graph.animate.shift(LEFT * 3.5),
            *[dot.animate.shift(LEFT * 3.5) for dot in self.dots],
        )
        # self.add(self.graph, *self.dots)
        return

    def construct_text(self):
        self.text_1 = (
            TexText(
                """
            Lastly, choose one data point from each group and select \\\\
            its associated frame in the video.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        self.text_2 = (
            TexText(
                """
            These keyframes can then be manually labeled for specific body parts.
            Here, the antennae, head, abdomen, thorax, and legs have been labeled.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        self.text_3 = (
            TexText(
                """
            The labeled keyframes are used to train pose estimation CNN models.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        self.text_4 = (
            TexText(
            """
            The trained models can be used to generate labels for entire videos.
            """
            )
            .scale(0.8)
            .to_edge(DOWN)
            .shift(DOWN * 0.2)
        )
        return


class twox(Scene):
    def construct(self):
        text = TexText("3x")
        speed = Speedometer(num_ticks=0)
        speed.needle.set_color(RED)
        speed.scale(text.get_height() / speed.get_height())
        group = VGroup(speed, text).arrange(RIGHT)
        group.to_edge(DOWN)
        self.play(Write(group[:]))
        self.wait(1)
        self.play(FadeOut(group))
        return


class KMeansAlgo(Scene):
    def initialize_parameters(self):
        """Initialize the parameters for the animation."""
        self.data = np.load("data/synth_data.npy")
        self.data = np.pad(self.data, ((0, 0), (0, 1)))  # manim needs 3D coordinates
        self.centroid_history = np.load("data/centroid_original_history.npy")
        self.centroid_history = np.pad(self.centroid_history, ((0, 0), (0, 0), (0, 1)))
        self.centroids = []

        self.DOT_COLOR = "#a7b8c7"
        self.C1_COLOR = "#00b4eb"
        self.C2_COLOR = "#40ff9f"
        self.C3_COLOR = "#ffa256"
        self.C4_COLOR = "#ff4d27"
        self.C_COLORS = [self.C1_COLOR, self.C2_COLOR, self.C3_COLOR, self.C4_COLOR]
        self.D1_COLOR = "#75cfeb"
        self.D2_COLOR = "#bfffdf"
        self.D3_COLOR = "#ffbe8a"
        self.D4_COLOR = "#ff8266"
        self.D_COLORS = [self.D1_COLOR, self.D2_COLOR, self.D3_COLOR, self.D4_COLOR]
        return

    def get_labels(self):
        """Return an array with the centroid labels for each dot based on proximity."""
        dot_positions = np.array([dot.get_center() for dot in self.dots])
        centroid_positions = np.array([c.get_center() for c in self.centroids])

        distances = cdist(dot_positions, centroid_positions)
        labels = np.argmin(distances, axis=1)

        return labels

    def construct(self):
        """Construct the scene."""
        self.initialize_parameters()
        self.construct_left_panel()
        self.construct_dividing_line()
        self.construct_right_panel()

        self.group_page()

        self.animate()
        return

    def animate(self):
        """Animate the scene."""
        # Write the title, line, and graph
        self.play(Write(self.title_group))
        self.play(Write(self.dividing_line), Write(self.graph))
        self.play(LaggedStart(*[FadeIn(dot) for dot in self.dots]), run_time=2)

        # Write step 1
        self.play(Write(self.step1_text[:]))
        self.construct_animate_centroid_init(0)

        # Write step 2
        self.play(Write(self.step2_text[:]))
        self.play(Write(self.step2_equation[:]))
        self.wait(0.5)
        self.construct_animate_centroid_distance_lines()

        self.play(Indicate(self.step2_text))
        self.wait(0.5)
        self.animate_label_assignment()

        # Write step 3
        self.play(
            Write(self.step3_text[:]),
            self.step2_equation[6].animate.set_color("#02BDCF"),
        )
        self.play(Write(self.step3_equation[:]))
        self.wait(0.75)

        self.animate_centroid_position_update(1)

        # Write step 4, iterate until convergence
        self.play(Write(self.step4_text[:]))
        for i in range(2, self.centroid_history.shape[0]):
            if i != 2:
                self.play(Indicate(self.step4_text, scale_factor=1.25))

            self.construct_animate_centroid_distance_lines(lag_ratio=0.025)

            self.play(Indicate(self.step2_text))
            self.wait(0.5)
            self.animate_label_assignment()

            self.play(Indicate(self.step3_text, scale_factor=1.5))
            self.wait(0.5)
            self.animate_centroid_position_update(i)

        self.animate_graph_centering()
        self.wait(0.75)

        self.animate_polygon_boundaries()
        self.wait(3)

        # Fade out remaining points
        self.fade_remaining()
        return

    def fade_remaining(self):
        """Fade the remaining objects in the scene."""
        self.play(
            FadeOut(self.title_group), FadeOut(self.graph_group), FadeOut(self.polygons)
        )
        return

    def animate_polygon_boundaries(self):
        """Animate boundaries of the centroids."""
        final_centroid_points = self.centroid_history[-1, :, :2]
        polygons_coords = get_polygons(final_centroid_points, [0, 5], [0, 5])

        print(len(polygons_coords))
        self.polygons = VGroup()
        for i, coords in enumerate(polygons_coords):
            polygon_coords = []
            for j in range(coords.shape[0]):
                polygon_coords.append(self.graph.coords_to_point(*coords[j]))
            polygon = Polygon(
                *polygon_coords,
                stroke_width=1.25,
                fill_opacity=0.3,
                fill_color=self.C_COLORS[i],
            )
            self.polygons.add(polygon)

        self.play(
            LaggedStart(*[Write(polygon) for polygon in self.polygons], lag_ratio=0.15),
            AnimationGroup(
                *[
                    Indicate(dot, color=dot.get_color(), scale_factor=1)
                    for dot in self.dots
                ]
            ),
            AnimationGroup(
                *[
                    Indicate(dot, color=dot.get_color(), scale_factor=1)
                    for dot in self.centroids
                ]
            ),
        )

        return

    def animate_graph_centering(self):
        """Center the graph after the convergence."""
        self.graph_group = VGroup(self.graph, *self.dots, *self.centroids)

        # self.title_replacement = self.title_group.copy()
        title_position = self.title_group.get_center_of_mass()
        title_position[0] = 0
        self.left_group.remove(self.title_group)

        self.play(
            self.title_group.animate.move_to(title_position),
            self.graph_group.animate.scale(1.25).move_to(ORIGIN).shift(DOWN * 0.2),
            *[FadeOut(object) for object in self.left_group],
            FadeOut(self.dividing_line),
        )
        return

    def animate_centroid_position_update(self, step):
        """Animate the update to the centroid positions."""
        new_positions = self.centroid_history[step]
        new_positions = [
            self.graph.coords_to_point(*position) for position in new_positions
        ]

        # First create centroid persistant animations
        def create_persistent_animations():
            animations = []  # keeps centroids above lines
            for i, centroid in enumerate(self.centroids):

                color_animation = centroid.animate.set_color(self.C_COLORS[i])
                animations.append(color_animation)

            return animations

        # Then animate lines from the dots to the new averaged position
        labels = self.get_labels()
        avg_lines = []
        for i, dot in enumerate(self.dots):
            label = labels[i]
            line = Line(
                dot.get_center(),
                new_positions[label],
                color=self.D_COLORS[label],
                stroke_width=1.5,
            )

            avg_lines.append(line)

        self.play(
            AnimationGroup(*[Write(line) for line in avg_lines]),
            AnimationGroup(*create_persistent_animations()),
        )  # play lines first
        self.wait(0.3)

        # then move the centroids to the new position
        centroid_animations = []
        for i, centroid in enumerate(self.centroids):
            animation = centroid.animate.move_to(new_positions[i])
            centroid_animations.append(animation)
        self.play(*centroid_animations)
        self.wait(0.3)

        # Lastly, get rid of the lines, animate them all into the new centroid
        animations = []
        for line in avg_lines:
            animation = line.animate.put_start_and_end_on(
                line.get_end(), line.get_end()
            )
            animations.append(animation)

        self.play(
            AnimationGroup(*animations), AnimationGroup(*create_persistent_animations())
        )
        self.remove(*avg_lines)
        self.wait(0.5)

        return

    def animate_label_assignment(self):
        """Animate the assignment of the labels to the dots."""
        # Retrieve the labels again
        labels = self.get_labels()

        # Animate the color shift of all of the labels
        dot_animations = []
        for i, dot in enumerate(self.dots):
            animation = dot.animate.set_color(self.D_COLORS[labels[i]])
            dot_animations.append(animation)

        # Then animate the projection of the final lines into each dot.
        # The "end" of each line should be at the dots because of how they were created
        line_animations = []
        for line in self.final_lines:
            animation = line.animate.put_start_and_end_on(
                line.get_end(), line.get_end()
            )
            line_animations.append(animation)

        # reanimate the centroids just to keep them on top
        centroid_animations = []
        for i, centroid in enumerate(self.centroids):
            centroid_animations.append(centroid.animate.set_color(self.C_COLORS[i]))

        self.play(
            LaggedStart(
                AnimationGroup(*line_animations),
                AnimationGroup(*dot_animations),
                AnimationGroup(*centroid_animations),
            ),
            run_time=2,
        )

        # Fade out the lines
        self.play(*[FadeOut(line) for line in self.final_lines])

        return

    def construct_animate_centroid_distance_lines(
        self, lag_ratio=0.065, indicate_min=True
    ):
        """Animate the line projections/retractions of the shortest lines to dots."""
        # Get the labels for the dots
        temp_lines = []
        self.final_lines = []

        labels = self.get_labels()

        # Animate the construction of all of the lines
        line_animations = []
        for i, dot in enumerate(self.dots):
            dot_line_animations = []
            for j, centroid in enumerate(self.centroids):
                line = Line(
                    centroid.get_center(),
                    dot.get_center(),
                    stroke_width=1.5,
                    color=self.C_COLORS[j],
                )
                dot_line_animations.append(Write(line))

                if j != labels[i]:
                    temp_lines.append(line)
                else:
                    self.final_lines.append(line)

            line_animations.append(AnimationGroup(*dot_line_animations))

        self.play(LaggedStart(*line_animations, lag_ratio=lag_ratio))

        # Animate the retraction of the shortest lines
        if indicate_min:
            self.play(Indicate(self.step2_equation[2], scale_factor=1.5))
            self.wait(0.5)
        self.play(*[Uncreate(line) for line in temp_lines])
        self.wait(0.4)
        return

    def construct_animate_centroid_init(self, step):
        """Construct and animate the creation of the selected centroids."""
        # Construct and then indicate all of the centroids for step 1
        self.centroids = []
        self.original_dots = []
        for i in range(self.centroid_history.shape[1]):
            # Construct the centroid
            coords = self.graph.coords_to_point(*self.centroid_history[step, i])

            centroid = Dot(coords, radius=0.1, color=self.C_COLORS[i])
            self.centroids.append(centroid)

            # Identify the first dot for the animation
            original_dot_index = np.argwhere(
                self.data[:, :2] == self.centroid_history[step, i, :2]
            )[0, 0]
            original_dot = self.dots[original_dot_index]
            self.original_dots.append(original_dot)

        animations = []
        for centroid, dot in zip(self.centroids, self.original_dots):
            animation = LaggedStart(
                Indicate(dot, scale_factor=6, color=centroid.get_color()),
                FadeIn(centroid),
            )
            animations.append(animation)

        self.play(LaggedStart(*animations, lag_ratio=0.5))
        return

    def group_page(self):
        """Group the main elements of the page together."""
        self.page_group = VGroup(self.left_group, self.dividing_line, self.right_group)
        self.page_group.arrange(RIGHT, buff=0.4)

        self.page_group.to_edge(LEFT)
        self.right_group.shift(RIGHT * 0.75)

        return

    def construct_graph(self):
        """Construct the graph and group the dots."""
        self.graph = Axes([0, 5], [0, 5], width=7.14, height=4.64, tick_size=0.05)
        self.graph.get_axes().set_stroke(width=3, color=WHITE)

        # update position of dots
        for i, dot in enumerate(self.dots):
            dot.move_to(self.graph.coords_to_point(*self.data[i]))

        # Create graph group
        self.graph_group = VGroup()
        self.graph_group.add(self.graph, *self.dots)

        return

    def construct_dots(self):
        """Construct the dots."""
        # Convert the input data to dots
        # Keep the radius small
        self.dots = [
            Dot(radius=0.05, color=self.DOT_COLOR) for _ in range(self.data.shape[0])
        ]
        return

    def construct_right_panel(self):
        """Construct the right panel.

        This function calls several separate functions for the graph creation.
        """
        self.construct_dots()
        self.construct_graph()

        self.right_group = VGroup(self.graph_group)
        return

    def construct_dividing_line(self):
        """Construct the dividing line."""
        self.dividing_line = Line(
            self.left_group.get_top(), self.left_group.get_bottom()
        )
        return

    def construct_left_panel(self):
        """Construct the left panel.

        This panel contains the scene title, the steps, and the equations.
        """
        # Title
        self.title = Text("K-Means Algorithm")
        self.title_ul = Underline(self.title)
        self.title_group = VGroup(self.title, self.title_ul)

        # Step 1
        self.step1_text = TexText("1. Initialize centroid \\\\" "positions (${k = 4}$)")

        # Step 2
        to_isolate = ["${\\mu}$", "${X}$"]
        self.step2_text = TexText(
            "2. Assign labels (${\mu}$) \\\\",
            "to all data (${X}$)",
            isolate=[*to_isolate],
        )
        self.step2_text.set_color_by_tex_to_color_map(
            {"${\\mu}$": "#DB238B", "${X}$": self.DOT_COLOR}
        )

        to_isolate = ["\\mu_{n}", "x_{n}", "c_{i}", "arg\\underset{i}{min}"]
        self.step2_equation = Tex(
            "\\mu_{n} = arg\\underset{i}{min}||x_{n} - c_{i}||", isolate=[*to_isolate]
        ).scale(0.9)
        self.step2_equation.set_color_by_tex_to_color_map(
            {"\\mu_{n}": "#DB238B", "x_{n}": self.DOT_COLOR}
        )
        step2_group = VGroup(self.step2_text, self.step2_equation).arrange(DOWN)

        # Step 3
        to_isolate = ["${C}$", "${\\mu}$"]
        self.step3_text = TexText(
            "3. Update centroid\\\\", "positions (${C}$)", isolate=[*to_isolate]
        )
        self.step3_text.set_color_by_tex_to_color_map({"${C}$": "#02BDCF"})

        to_isolate = ["c_{i}", "x_{n}"]
        self.step3_equation = Tex(
            "c_{i} = {\\sum_{n=1}^{N} x_{n} \\over \\sum_{n=1}^{N}A_{nk}}",
            isolate=[*to_isolate],
        )
        self.step3_equation.set_color_by_tex_to_color_map(
            {"c_{i}": "#02BDCF", "x_{n}": self.DOT_COLOR}
        )
        step3_group = VGroup(self.step3_text, self.step3_equation).arrange(DOWN)

        # Step 4
        self.step4_text = TexText("4. Repeat steps 2 \& 3\\\\", "until convergence")

        # Group the panel
        self.left_group = VGroup(
            self.title_group,
            self.step1_text,
            step2_group,
            step3_group,
            self.step4_text,
        ).scale(0.8)
        self.left_group.arrange(DOWN, buff=0.375)

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
        self.C2_COLOR = "#40ff9f"
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

class Title(Scene):
    def construct(self):
        self.title = TexText("K-Means Clustering Algorithm").scale(0.8).to_edge(UP).shift(UP*0.2)
        
        line_left = LEFT_SIDE + [0.5, 0, 0]
        line_left[1] = self.title.get_bottom()[1] * 0.95
        line_right = RIGHT_SIDE - [0.5, 0, 0]
        line_right[1] = self.title.get_bottom()[1] * 0.95
        self.title_underline = Line(line_left, line_right, color=WHITE, stroke_width=2)
        self.title_group = VGroup(self.title, self.title_underline)
        
        self.play(Write(self.title_group[:]))
        return