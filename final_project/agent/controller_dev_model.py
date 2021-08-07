from time import time

import pystk, os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from . import gui
from .controller import Controller_A
from argparse import ArgumentParser
import numpy as np
from .controller import Controller_A
from .models import load_model
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])


def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = ArgumentParser("Play some Ice Hockey. List any number of players, odd players are in team 1, even players team 2.")
    parser.add_argument('--w', default=0, type=int, help="Who is the goalie?")
    args = parser.parse_args()
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300

    pystk.init(config)

    config = pystk.RaceConfig()
    config.track = "icy_soccer_field"
    config.mode = config.RaceMode.SOCCER
    config.step_size = 0.1
    config.num_kart = 4
    config.players[0].kart = "wilber"
    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    config.players[0].team = 0
    config.players.append(pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1))
    config.players.append(pystk.PlayerConfig("", pystk.PlayerConfig.Controller.PLAYER_CONTROL, 0))
    config.players.append(pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1))
    config.difficulty = 2
    race = pystk.Race(config)
    race.start()

    model = load_model()

    uis = [gui.UI([gui.VT['IMAGE']])]

    state = pystk.WorldState()
    t0 = time()
    n = 0

    ax = plt.gcf().add_subplot(3, 3, 9)
    
    # Hard coded goal line
    goal_line = np.array([[[-10.449999809265137, 0.07000000029802322, -64.5], [10.449999809265137, 0.07000000029802322, -64.5]], [[10.460000038146973, 0.07000000029802322, 64.5], [-10.510000228881836, 0.07000000029802322, 64.5]]])
    
    
    # team_orientaion_multiplier = -2*(config.players[0].team %2)+1
    ctrl0 = Controller_A(config.players[0].team,0)
    ctrl1 = Controller_A(config.players[0].team,2)
    last_seen_side0 = None
    last_seen_side1 = None
    goal = np.array([0.0,64.5])

    race.step(uis[0].current_action)
    state.update()
    while all(ui.visible for ui in uis):
        if not all(ui.pause for ui in uis):
            #print(uis[0].current_action)
            race.step(uis[0].current_action)
            #race.step(uis[1].current_action)
            state.update()
            #race.step(uis[0].current_action)
            #state.update()
            
        image = np.array(race.render_data[0].image)
        print(state.players[0])
        player_info = state.players[0]
        real_puck, real_loc, world_loc, last_see = model.detect(to_tensor(image), player_info)
        # last_seen_side0 = last_see

        if real_puck== False:
            puck_location = None
        else:
            puck_location = world_loc
        print(f"detected loc: {puck_location}")
        
        puck_location_gt = to_numpy(state.soccer.ball.location) # We need to get this from NN output
        print(f"detected gt: {puck_location_gt}")
        
        # if (len(state.karts)==1):
        #     pos_ai =np.array([0.0,0.0])
        # else:
        #     pos_ai = to_numpy(state.karts[1].location)
        # pos_me = to_numpy(state.karts[0].location)


        #  Standardizing direction 2 elements
        # [0] is negitive when facing left side of court (left of your goal), positive when right
        # [1] is positive towards enemy goal, negitive when facing your goal
        # if puck_location:
        #     puck_location*=team_orientaion_multiplier
        # pos_ai*=team_orientaion_multiplier
        # pos_me*=team_orientaion_multiplier


        # Look for the puck
        # closest_item_distance = np.linalg.norm(
        #             get_vector_from_this_to_that(pos_me, puck_location, normalize=False))

        # # Get some directional vectors. 
        # front_me = to_numpy(state.karts[0].front)*team_orientaion_multiplier
        # ori_me = get_vector_from_this_to_that(pos_me, front_me)
        # ori_to_ai = get_vector_from_this_to_that(pos_me, pos_ai)
        # if puck_location:
        # ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location)
        # ori_puck_to_goal = get_vector_from_this_to_that(puck_location, goal)
        #print("ori_puck_to_goal",ori_puck_to_goal)
        
        #otp = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
        #print("ori_to_puck",otp,np.linalg.norm(otp))


        # set actions for kart0
        action0 = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0, 'fire': False}
        # turn_mag = abs(1 - np.dot(ori_me, ori_to_puck))
        # if (turn_mag >.4):
        #     action0 = ctrl0.act(action0, state.karts[0],puck_location=None,last_seen_side=last_seen_side0, testing=True)
        # else:
        #     print("WE CAN SEE")
        #     last_seen_side0 = np.sign(np.cross(ori_to_puck, ori_me))

        action0 = ctrl0.act(state.karts[0], puck_location=puck_location,last_see=last_see, debug=False)
 
        uis[0].current_action.steer = action0["steer"]
        uis[0].current_action.acceleration = action0["acceleration"]
        uis[0].current_action.brake = action0["brake"]
        uis[0].current_action.fire = action0["fire"]
        uis[0].current_action.rescue = action0["rescue"]
        print(uis[0].current_action)
        
        # if (len(state.karts)>=3 and len(uis)>=3):
        #     pos_me1 = to_numpy(state.karts[0].location)*team_orientaion_multiplier
        #     front_me1 = to_numpy(state.karts[2].front)*team_orientaion_multiplier
        #     ori_me1 = get_vector_from_this_to_that(pos_me1, front_me1)
        #     ori_to_puck1 = get_vector_from_this_to_that(pos_me1, puck_location)
        #     action1 = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0, 'fire': False}
        #     turn_mag1 = abs(1 - np.dot(ori_me1, ori_to_puck1))
        #     if (turn_mag1 >.4):
        #         action1 = ctrl1.act(action1, state.karts[2],puck_location=None,last_seen_side=last_seen_side1, testing=True)
        #     else:
        #         print("WE CAN SEE")
        #         last_seen_side1 = np.sign(np.cross(ori_to_puck, ori_me))
        #         action1 = ctrl1.act(action1, state.karts[2], puck_location=puck_location,last_seen_side=last_seen_side1,testing=True)
    
        #     uis[2].current_action.steer = action1["steer"]
        #     uis[2].current_action.acceleration = action1["acceleration"]
        #     uis[2].current_action.brake = action1["brake"]
        #     uis[2].current_action.fire = action1["fire"]


        
        #race.step(uis[0].current_action)
        #state.update()
        

        # Live plotting. Sorry it's ugly.
        # ax.clear()
        # ax.set_xlim(-100, 100)
        # ax.set_ylim(-100, 100)

        # ax.plot(pos_me[0], pos_me[1], 'r.')                 # Current player is a red dot.
        # ax.plot(pos_ai[0], pos_ai[1], 'b.')                 # Enemy ai is a blue dot.
        # ax.plot(puck_location[0], puck_location[1], 'co')             # The puck is a cyan circle.
        # ax.plot(puck_location[0], puck_location[1], 'kx')   # The target picked up is a black x.

        # # Plot lines of where I am facing, and where the enemy is in relationship to me.
        # ax.plot([pos_me[0], pos_me[0] + 10 * ori_me[0]], [pos_me[1], pos_me[1] + 10 * ori_me[1]], 'r-')
        # ax.plot([pos_me[0], pos_me[0] + 10 * ori_to_ai[0]], [pos_me[1], pos_me[1] + 10 * ori_to_ai[1]], 'b-')
        # ax.plot([puck_location[0], puck_location[0] + 10 * ori_puck_to_goal[0]], [puck_location[1], puck_location[1] + 10 * ori_puck_to_goal[1]], 'b-')

        # # Live debugging of scalars. Angle in degrees to the target item.
        # ax.set_title('%.2f' % (np.degrees(np.arccos(np.dot(ori_me, ori_to_puck)))))

        # Properties of the karts. Overall useful to see what properties you have.
        # print(dir(state.karts[0]))
        print(len(race.render_data))
        if (len(uis)>=3):
            #for ui, d in zip(uis, race.render_data):
            #    ui.show(d)
            for ui, d in zip([uis[1],uis[2]], [race.render_data[1],race.render_data[2]]):
                ui.show(d)
        else:
            
            for ui, d in zip(uis, race.render_data):                
                if real_puck:
                    x,y = real_loc
                    ui.show(d, x, y)
                else:
                    ui.show(d, x=None, y=None)
                # d = cv2.rectangle(d, (x-20,y-20), (x+20,y+20), color=(0,255,0), thickness=4)
        

        # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time() - t0)
        if delta_d > 0: ui.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
