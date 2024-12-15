import cv2

# Function to read all frames from a video file
def read_video(video_path):
    """
    Reads a video file and extracts all its frames.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        frames (list): List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Stop if no more frames are available
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames into a video file.
    Args:
        output_video_frames (list): List of frames to save.
        output_video_path (str): Path to the output video file.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Video codec for output
    height, width, _ = output_video_frames[0].shape # Frame dimensions
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width,height)) # Create VideoWriter
    for frame in output_video_frames:
        frame = cv2.resize(frame, (width, height)) # Ensure frame dimensions match output
        out.write(frame)
    out.release()