import numpy as np

# Calculates the center of a bounding box
def get_center_bbox(bbox):
    """
    Computes the center coordinates of a bounding box.
    Args:
        bbox (tuple): Bounding box as (x1, y1, x2, y2).
    Returns:
        tuple: Center coordinates (x_center, y_center).
    """
    x1,y1,x2,y2 = bbox
    center = int((x1+x2)/2), int((y1+y2)/2)
    return center

# Calculates the width of a bounding box
def get_width_bbox(bbox):
    """
    Computes the width of a bounding box.
    Args:
        bbox (tuple): Bounding box as (x1, y1, x2, y2).
    Returns:
        int: Width of the bounding box.
    """
    width = bbox[2]-bbox[0]
    return width

# Calculates the height of a bounding box
def get_height_bbox(bbox):
    """
    Computes the height of a bounding box.
    Args:
        bbox (tuple): Bounding box as (x1, y1, x2, y2).
    Returns:
        int: Height of the bounding box.
    """
    height = bbox[3]-bbox[1]
    return height

# Finds a bounding box within team information
def find_bbox_on_team(teams_info, box):
    """
    Searches for a bounding box within the team information.
    Args:
        teams_info (dict): Information about teams and players.
        box (array): Bounding box to search for.
    Returns:
        dict or None: Details about the frame, team, and player if found, otherwise None.
    """
    for frame_num, teams in teams_info.items():
        for team, jugadores in teams.items():
            for jugador in jugadores:
                if np.array_equal(jugador['bbox'], box):
                    return {'frame':frame_num, 'team': team, 'jugador': jugador}
    return None
    