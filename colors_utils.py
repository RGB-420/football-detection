import cv2
import numpy as np
from sklearn.cluster import KMeans

# Converts an RGB color to HSV
def rgb_to_hsv(rgb_color):
    """
    Converts an RGB color to HSV color space.
    Args:
        rgb_color (tuple): Input RGB color as (R, G, B).
    Returns:
        hsv_color (tuple): Hue value in the HSV color space.
    """
    color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0][0]
    return color

# Converts an HSV color to RGB
def hsv_to_rgb(hsv_color):
    """
    Converts an HSV color to RGB color space.
    Args:
        hsv_color (tuple): Input HSV color.
    Returns:
        rgb_color (tuple): Converted RGB color.
    """
    rgb_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2RGB)[0][0]
    return rgb_color

# Computes the Euclidean distance between two colors
def color_distance(color1, color2):
    """
    Calculates the Euclidean distance between two colors.
    Args:
        color1 (tuple): First color in RGB space.
        color2 (tuple): Second color in RGB space.
    Returns:
        float: Euclidean distance between the two colors.
    """
    distance = np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))
    return distance

# Groups similar colors using K-Means clustering
def group_similar_colors(colors, num_clusters=2):
    """
    Groups similar colors into clusters using K-Means.
    Args:
        colors (list): List of RGB colors.
        num_clusters (int): Number of clusters to form.
    Returns:
        centers (list): RGB centers of the clusters.
    """
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(colors)
    
    # Get cluster centers
    centers = kmeans.cluster_centers_.astype(int)
    return centers