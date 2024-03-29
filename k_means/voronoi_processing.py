import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.ops import clip_by_rect


def interpolate_centroid_history(
    centroids_history: list,
    frames: int,
    switch_buffer: int = 0,
):
    """Interpolate a centroid path at 30fps for a desired frame count."""
    # get the number of frames needed between each keyframe
    frame_counts = np.linspace(0, frames, len(centroids_history), dtype=np.int_)
    frame_counts = frame_counts[1:] - frame_counts[:-1]
    
    # interpolate the frames between each keyframe and add to the list
    interpolated_history = []
    for i in range(len(centroids_history) - 1):
        if switch_buffer:
            interpolated_history.extend(
                np.repeat([centroids_history[i]], switch_buffer, axis=0)
            )

        interpolated_centroids = np.linspace(
            centroids_history[i], centroids_history[i + 1], frame_counts[i]
        )
        interpolated_history.extend(interpolated_centroids)

    interpolated_history = np.array(interpolated_history)
    return interpolated_history


def clip_polygon(polygon: np.ndarray, x_range: list, y_range: list):
    """Clip a polygon inside of an x and y input range."""
    poly = Polygon(polygon)
    clipped = clip_by_rect(poly, x_range[0], y_range[0], x_range[1], y_range[1])
    clipped = np.array([p for p in clipped.exterior.coords.xy]).T
    return clipped


def check_intersection(polygon: np.ndarray, bounding_box: np.ndarray):
    """Return whether the input polygon intersects the bounding box viewport."""
    poly = Polygon(polygon)
    bounding_poly = Polygon(bounding_box)
    intersection = poly.intersection(bounding_poly)
    return True if intersection else False


def generate_boundary_points(centroids: np.ndarray) -> np.ndarray:
    """Generate boundary points to ensure polygon completion from a voronoi algo."""
    # Populate the plane with other points to avoid polygon construction issues.
    line1 = np.linspace((-1000, 1000), (1000, 1000), 20)
    line2 = np.linspace((1000, 1000), (1000, -1000), 20)
    line3 = np.linspace((1000, -1000), (-1000, -1000), 20)
    line4 = np.linspace((-1000, -1000), (-1000, 1000), 20)
    line5 = np.linspace((-1000, 0), (0, 1000), 20)
    line6 = np.linspace((0, 1000), (1000, 0), 20)
    line7 = np.linspace((1000, 0), (0, -1000), 20)
    line8 = np.linspace((0, -1000), (-1000, 0), 20)  # 160 pts is overkill but whatever
    lines = [line1, line2, line3, line4, line5, line6, line7, line8]
    coords = np.concatenate([centroids, *lines], axis=0)
    return coords


def generate_bounding_box(x_range: list, y_range: list) -> np.ndarray:
    """Return a bounding box from the input axes ranges."""
    bounding_box = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
    return bounding_box[[0, 1, 3, 2]]


def get_polygons(centroids: np.ndarray, x_range: list, y_range: list) -> list:
    """Return a list of the voronoi polygons associated with each centroid."""
    # Add some outlier boundary points to the plot in order to fill in the voronoi
    coords = generate_boundary_points(centroids)

    # Create our bounding box array from the input x and y ranges.
    bounding_box = generate_bounding_box(x_range, y_range)

    # Create the voronoi diagram
    vor = Voronoi(coords)

    # Iterate through the region associated with each input point and create a polygon
    polygons = []
    for r in range(coords.shape[0]):
        region = vor.regions[vor.point_region[r]]  # get the point region

        # make sure the region has a complete polygon
        if -1 in region:
            continue

        polygon = np.array([vor.vertices[i] for i in region])

        # If the polygon intersects with our bounding box, add it to the list
        if check_intersection(polygon, bounding_box):
            polygon = clip_polygon(polygon, x_range, y_range)  # clip to the box
            polygons.append(polygon)

    return polygons
