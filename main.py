from utils import read_video, save_video
from tracker import Tracker

def main():
    # Read video
    video_frames = read_video('Videos\Prueba_Corta.mp4')

    # Initialize Tracker
    tracker = Tracker('train/models/Best.pt')

    # Get object tracks
    tracks = tracker.get_objects_track(video_frames)

    #Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_video.avi')

if __name__ == '__main__':
    main()