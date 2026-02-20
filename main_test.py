from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from utils import read_video, save_video

def test_main():
 # Read Video
    video_frames = read_video('input_videos/vedio_test.mp4') 
    
    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks =  tracker.get_object_tracks(
        video_frames,
        read_from_stub= True,
        stub_path= 'stubs/track_stubs.pkl'
    )

 # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')