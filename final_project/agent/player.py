import numpy as np
from .controller import Controller_A
from .models import load_model
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import time

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """
    
    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = "wilber"
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        team_id = player_id % 2
        self.controller = Controller_A(team_id, player_id)
        self.model = load_model()

    def act(self, image, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        """
        Your code here.
        """
        
        start_time = time.time()*1000
        real_puck, real_loc, world_loc, last_see = self.model.detect(to_tensor(image), player_info)
        detect_endtime = time.time()*1000
        if real_puck == False:
            world_loc = None

        action = self.controller.act(puck_location=world_loc, player_info=player_info, last_see=last_see)
        print(f"model time: {detect_endtime-start_time}")
        print(f"controller time: {time.time()*1000-detect_endtime}")

        return action