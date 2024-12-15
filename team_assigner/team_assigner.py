from sklearn.cluster import KMeans
from utils import group_similar_colors, color_distance

class TeamAssigner:
    def __init__(self):
        # Dictionary to store team colors
        self.team_colors = {}
        # Information about teams assigned to each player
        self.teams_info = {}
        # Referee's assigned color
        self.referee_color = ()

    def get_clustering_model(self, image):
        """
        Creates and fits a K-Means clustering model for an image.
        Args:
            image: Input image as a NumPy ndarray.
        Returns:
            kmeans: Fitted K-Means model.
        """
        # Reshape the image into a 2D array (number of pixels x 3 color channels)
        image_2d = image.reshape(-1,3)

        # Configure K-Means with 2 clusters (default)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant color of a player within a bounding box.
        Args:
            frame: Current frame as a NumPy ndarray.
            bbox: Bounding box coordinates (x1, y1, x2, y2).
        Returns:
            player_color_bgr: Player's color in BGR format.
        """
        # Extract the region of interest (ROI) from the image using the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Focus on the central region to avoid background or edge noise
        center_image = image[int(image.shape[0]*0.25):int(image.shape[0]*0.75),int(image.shape[1]*0.1):int(image.shape[1]*0.9)]

        # Get the clustering model for the central region
        kmeans = self.get_clustering_model(center_image)
        labels = kmeans.labels_

        # Reshape cluster labels back to image dimensions
        clustered_image = labels.reshape(center_image.shape[0],center_image.shape[1])

        # Identify the non-player cluster based on corner pixels
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count) 
        player_cluster = 1 - non_player_cluster # The other cluster represents the player

        # Get the color of the player's cluster center
        player_color = kmeans.cluster_centers_[player_cluster]

        # Convert to BGR format and ensure valid pixel values (0-255)
        player_color_bgr = tuple(map(lambda x: min(max(int(round(x)), 0), 255), player_color[::-1]))

        return player_color_bgr
    
    def get_team_colors(self, frame, tracks):
        """
        Identifies team colors from detected players.
        Args:
            frame: Frame where the players are detected.
            tracks: Tracking data with bounding boxes for players.
        Returns:
            team_colors: A list of team colors.
        """
        # List to store all player colors
        player_colors = []

        # Check if tracking data is available
        if not tracks:
            print("No tracking data available.")
            return []
        # Get the results from the first frame
        first_frame_results = tracks[0]

        # Check if 'boxes' attribute exists and is not None in the first frame
        if not any(hasattr(results, 'boxes') and results.boxes is not None for results in first_frame_results):
            print("No se encontraron detecciones para el primer frame.")
            return []

        # Iterate through the results in the first frame
        for results in first_frame_results:
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.numpy()
                class_ids = results.boxes.cls.numpy().astype(int)

                # Process each detected player
                for box, class_id in zip(boxes, class_ids):
                    if class_id == 2: # Class ID 2 represents players
                        player_color = self.get_player_color(frame, box)
                
                        # Ensure player_color is a tuple of three integers
                        if isinstance(player_color, (list, tuple)) and len(player_color) == 3 and all(isinstance(c, int) for c in player_color):
                            player_colors.append(tuple(player_color))
                        else:
                            print(f"Warning: Invalid player color detected: {player_color}. Using default color (0, 0, 0).")
                            player_colors.append((0, 0, 0))

        # Group similar colors to identify team colors
        team_colors = group_similar_colors(player_colors)

        # Store the team colors in the class
        self.team_colors = team_colors

        return team_colors

    def get_referee_color(self,frame,box):
        """
        Determines the referee's color based on their bounding box.
        Args:
            frame: Current frame as a NumPy ndarray.
            box: Bounding box for the referee.
        Returns:
            referee_color: Referee's color in BGR format.
        """
        referee_color = self.get_player_color(frame,box)

         # Store the referee color in the class
        self.referee_color = referee_color

        return referee_color


    def get_team_for_player(self,player_color):
        """
        Assigns a player to the closest team based on color distance.
        Args:
            player_color: Color of the player in BGR format.
        Returns:
            team_id: 1 for Team 1, 2 for Team 2.
        """
        # Compute the color distance to each team's color
        dist_team_1 = color_distance(player_color, self.team_colors[0])
        dist_team_2 = color_distance(player_color, self.team_colors[1])

        # Assign the player to the team with the smallest distance
        if dist_team_1 < dist_team_2:
            return 1
        else:
            return 2
        
    def assign_team(self, frame, player_box):
        """
        Determines the team ID for a specific player.
        Args:
            frame: Current frame as a NumPy ndarray.
            player_box: Bounding box for the player.
        Returns:
            team_id: 1 for Team 1, 2 for Team 2.
        """
        player_color = self.get_player_color(frame, player_box)
        team_id = self.get_team_for_player(player_color)
        return team_id
    
    def get_teams(self, video_frames, tracks):
        """
        Assigns players to teams for each frame in a video sequence.
        Args:
            video_frames: List of video frames as NumPy arrays.
            tracks: Tracking data with bounding boxes for each frame.
        Returns:
            teams: Dictionary containing team assignments for each frame.
        """
        teams = {}

        # Iterate through each frame
        for frame_num, frame in enumerate(video_frames):

            results_list = tracks[frame_num]

            teams[frame_num] = {"team_1":[], "team_2": []}

            # Process detected players in the frame
            for results in results_list:
                # Check if 'boxes' attribute exists
                if not hasattr(results, 'boxes') or results.boxes is None:
                    print(f"No detections found for frame {frame_num}.")
                    continue

            boxes = results.boxes.xyxy.numpy()
            class_ids = results.boxes.cls.numpy().astype(int)

            for box, class_id in zip(boxes, class_ids):
                if class_id in [2]: # Class ID 2 represents players
                    team = self.assign_team(frame,box)
                    player_info = {"bbox": box.tolist()}
                    if team == 1:
                        teams[frame_num]["team_1"].append(player_info)
                    else:
                        teams[frame_num]["team_2"].append(player_info)

        # Store the team information in the class
        self.teams_info = teams

        return teams

