import sys

from utils.bbox_utils import get_foot_position
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 50 

    def assign_ball_to_player(self, players, ball_bbox):

     ball_position = get_center_of_bbox(ball_bbox)

     minimum_distance = float("inf")
     assigned_player = -1

     for player_id, player in players.items():

        player_bbox = player['bbox']

        # utiliser la vraie position du pied
        player_position = get_foot_position(player_bbox)

        distance = measure_distance(player_position, ball_position)

        if distance < self.max_player_ball_distance:
            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

     return assigned_player
