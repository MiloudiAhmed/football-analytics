import pickle  
import cv2  
import numpy as np 
import os
import sys

sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():  

    def __init__(self, frame): 
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        ) 

        first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        mask = np.zeros_like(first_gray)  
        mask[:, 0:20] = 1  
        mask[:, 900:1050] = 1  

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask
        ) 

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object_name][frame_num][track_id]['position_adjusted'] = position_adjusted


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        if old_features is None:
            return camera_movement

        for i in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue

            max_dist = 0
            dx, dy = 0, 0

            for new, old in zip(new_features, old_features):
                new_pt = new.ravel()
                old_pt = old.ravel()

                dist = measure_distance(new_pt, old_pt)
                if dist > max_dist:
                    max_dist = dist
                    dx, dy = measure_xy_distance(old_pt, new_pt)

            if max_dist > self.minimum_distance:
                camera_movement[i] = [dx, dy]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement    

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for i, frame in enumerate(frames):
            
            overlay = frame

            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            if i < len(camera_movement_per_frame):
                x, y = camera_movement_per_frame[i]
            else:
                x, y = 0, 0


            cv2.putText(frame, f"Camera X: {x:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            cv2.putText(frame, f"Camera Y: {y:.2f}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            output_frames.append(frame)

        return output_frames
