from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
import concurrent.futures
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
def main():  
    # Read Video
    video_frames = read_video('input_videos/vedio_test.mp4') 
    
    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Run heavy tasks (tracking + camera movement) in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        tracks_future = executor.submit(
            tracker.get_object_tracks,
            video_frames,
             True,   # maintenant True
            'stubs/track_stubs.pkl'
        )

        camera_future = executor.submit(
            camera_movement_estimator.get_camera_movement,
            video_frames,
            True,
            'stubs/camera_movement_stub.pkl'
        )

        # Wait for both results
        tracks = tracks_future.result()
        camera_movement_per_frame = camera_future.result()
    
    #Get object positions
    tracker.add_position_to_tracks(tracks)
 
    num_frames = len(tracks['players'])
    if len(camera_movement_per_frame) < num_frames:
    # Répéter le dernier mouvement pour compléter
       last_movement = camera_movement_per_frame[-1]
       for _ in range(num_frames - len(camera_movement_per_frame)):
           camera_movement_per_frame.append(last_movement)
    elif len(camera_movement_per_frame) > num_frames:
        camera_movement_per_frame = camera_movement_per_frame[:num_frames]


    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Transformer 
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and Distance Estimation
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams  
    team_assigner = TeamAssigner()  
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])  
    
    for frame_num, player_track in enumerate(tracks['players']):  
        for player_id, track in player_track.items():  
            team = team_assigner.get_player_team(
                video_frames[frame_num],  
                track['bbox'],  
                player_id
            )  
            tracks['players'][frame_num][player_id]['team'] = team  
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        if (frame_num < len(tracks['ball']) and 
            tracks['ball'][frame_num] and 
            1 in tracks['ball'][frame_num]):
            
            ball_bbox = tracks['ball'][frame_num][1]['bbox']     
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(None)
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)
                
    team_ball_control = np.array(team_ball_control)

    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, 
        camera_movement_per_frame
    )

    # Draw Speed and Distance 
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':  
    main()