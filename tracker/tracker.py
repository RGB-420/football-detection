from ultralytics import YOLO
import numpy as np
import sys
import cv2

# Append the parent directory to sys.path to import custom utility modules
sys.path.append('../')
from utils import get_center_bbox, get_width_bbox, get_height_bbox, find_bbox_on_team
from team_assigner import TeamAssigner

# Initialize the TeamAssigner class
team_assigner = TeamAssigner()

class Tracker:
    def __init__(self,model_path):
        """
        Initializes the Tracker with a YOLO model.
        Args:
            model_path: Path to the YOLO model.
        """
        self.model = YOLO(model_path)

    def get_objects_track(self,frames):
        """
        Tracks objects across multiple frames using YOLO.
        Args:
            frames: List of video frames to analyze.
        Returns:
            all_results: List of YOLO tracking results for each frame.
        """
        all_results = []

        for frame in frames:
            # Run the YOLO model in tracking mode with a confidence threshold
            results = self.model.track(frame,persist=True,conf=0.1)

            all_results.append(results)

        return all_results
    
    def draw_ellipse(self,frame,bbox,color):
        """
        Draws an ellipse on the frame to annotate players, referees, or other objects.
        Args:
            frame: Video frame as a NumPy array.
            bbox: Bounding box (x1, y1, x2, y2) of the object.
            color: Color of the ellipse in (B, G, R) format.
        """
        # Get the bottom y-coordinate of the bounding box
        y2 = int(bbox[3])

        # Calculate the center and width of the bounding box
        x_center, _ = get_center_bbox(bbox)
        width = get_width_bbox(bbox)

        # Ensure the color is a valid BGR tuple
        if not isinstance(color, tuple) or len(color) != 3 or not all(isinstance(c, int) for c in color):
            raise ValueError(f"Invalid color: {color}. Expected a tuple of 3 integers (B, G, R).")

        # Draw an ellipse around the bounding box
        cv2.ellipse(frame,
                    center = (x_center,y2), 
                    axes = (int(width), int(width*0.35)), 
                    angle=180, startAngle=-240, endAngle=60,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                    )

    def set_color(self, frame, class_id, box):   
        """
        Determines the color to annotate objects based on their class ID.
        Args:
            frame: Current video frame.
            class_id: Class ID of the detected object.
            box: Bounding box of the detected object.
        Returns:
            color: BGR color for annotation.
        """      

        if class_id == 3: # Referee
            if not first_referee:
                color = team_assigner.get_referee_color(frame,box) # Assign color for the referee if it's the first occurrence
                first_referee = True
            else:
                # Use previously assigned referee color
                color = team_assigner.referee_color

        elif class_id == 1: # Goalkeeper
            color = team_assigner.get_player_color(frame,box)

        elif class_id == 2: # Player
            # Find the player in team data using their bounding box            
            player = find_bbox_on_team(team_assigner.teams_info,box)
            if player == None:
                print("Error 424: Player Not Found")
            else:
                # Assign color based on the team
                team = player['team']
                if team == 'team_1':
                    color = team_assigner.team_colors[0]
                elif team == 'team_2':
                    color = team_assigner.team_colors[1]
                else:
                    print("No team assigned, LOL, must be bad at playing.")
        
        return color

    def draw_annotations(self,video_frames, tracks):
        """
        Annotates video frames with team and object data.
        Args:
            video_frames: List of video frames.
            tracks: Object tracking data from YOLO.
        Returns:
            output_video_frames: List of annotated video frames.
        """
        output_video_frames = []

        # Extract team colors and team assignments
        team_assigner.get_team_colors(video_frames[0],tracks)
        team_assigner.get_teams(video_frames,tracks)

        # Iterate over each frame
        for frame_num, frame in enumerate(video_frames):
            # Make a copy of the frame to preserve the original
            frame = frame.copy()

            # Get tracking results for the current frame
            results_list = tracks[frame_num]

            for results in results_list:
                # Verify that boxes exist in the tracking results
                if not hasattr(results, 'boxes') or results.boxes is None:
                    print(f"No detections found for frame {frame_num}.")
                    continue
                
                # Extract bounding boxes and class IDs from the results
                boxes = results.boxes.xyxy.numpy()
                class_ids = results.boxes.cls.numpy().astype(int)

                # Annotate detected objects
                for box, class_id in zip(boxes, class_ids):
                
                    if class_id in [ 1, 2, 3]:  # 0: Ball, 1: GK, 2: Player, 3: Referee
                        
                        # Get the annotation color
                        color = self.set_color(frame,class_id,box)
                        color_depurated = tuple(map(int, color[::-1]))

                        # Draw an ellipse for the object
                        self.draw_ellipse(frame, box, color_depurated)

            # Append the annotated frame to the output
            output_video_frames.append(frame)

        return output_video_frames
