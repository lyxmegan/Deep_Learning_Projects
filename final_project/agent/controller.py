import numpy as np
import torch
DEBUG = True
DEFEND_STEER_RATIO = 1
ATTACK_STEER_RATIO = 8000

class Puck_location_history:
    def __init__(self):
        self.queue = []
        self.max_length = 10
        self.init_value = np.array([0.0,0.0])
    
    def __len__(self):
        return len(self.queue)

    def push(self, e):
        if len(self.queue) >0:
            if e[1] - self.queue[-1][1] < 0:
                if len(self.queue) < self.max_length:
                    self.queue.append(e)
                else:
                    self.queue.pop(0)
                    self.queue.append(e)
        else:
            if len(self.queue) < self.max_length:
                    self.queue.append(e)
            else:
                self.queue.pop(0)
                self.queue.append(e)
    
    def get_history(self, N=1):
        if (len(self.queue)>=N):
            return self.queue[-N:]

        return self.init_value

def to_numpy(location):

    return np.float32([location[0], location[2]])

def get_vector_from_this_to_that(me, obj, normalize=True):
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def calculate_turn_angle(unit_vec1, unit_vec2):
    return abs(1 - np.dot(unit_vec1, unit_vec2))


class Controller_A:    

    def __init__(self, team_id, player_id):
        """
        :param team: A 0 or 1 representing which team we are on
        """
        goal_line = 64.5

        if team_id == 0:
            self.team_direction = 1
        else: # team_id == 1
            self.team_direction = -1

        self.player_id = player_id
        self.goal = np.array([0.0, goal_line])
        self.guard_loc = np.array([0.0, -67])
        self.his_buffer = Puck_location_history()
        self.last_position = np.array([0.0, -goal_line])
        self.rescue_count = 1
        self.last_see_position = np.array([0.0, 0.0])
        self.center = np.array([0.0, 0.0])
        self.last_see_count = 1
     
    def act(self, player_info, puck_location=None, last_see=-1, debug=False):

        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        action["fire"]= True
            
        if debug:
            if (puck_location is not None):
                puck_location*=self.team_direction
            kart = player_info
        else:
            kart = player_info.kart
        pos_me = to_numpy(kart.location)*self.team_direction
        
        # Get kart vector
        front_me = to_numpy(kart.front)*self.team_direction
        ori_me = get_vector_from_this_to_that(pos_me, front_me)

        # Determine we are moving backwards
        driving_direction = 1.
        kart_vel = np.dot(to_numpy(kart.velocity)*self.team_direction, ori_me)
        driving_direction = -1. if kart_vel < 0 else 1.
        # print(f"id: {self.player_id}")

        if  self.player_id // 2 == 0: 
            # print("Attacker")
            if puck_location is not None:
                self.last_see_count = 1
                self.last_see_position = puck_location
                self.rescue_count = 1
                ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location, normalize=True)
                ori_puck_to_goal_n = get_vector_from_this_to_that(puck_location, self.goal, normalize=True)

                to_puck_mag = np.linalg.norm(ori_to_puck)

                if (to_puck_mag >20): # not close to puck
                    action["acceleration"] = 1. #.8
                    action["nitro"] = True
                else:
                    action["acceleration"] = .9
                    if (to_puck_mag>10):# really close
                        action["acceleration"] = .8
                    pos_hit_loc = puck_location - 1.5 * ori_puck_to_goal_n
                    ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)

                turn_project = calculate_turn_angle(ori_me, ori_to_puck_n)
                if turn_project > 1e-25:
                    action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*turn_project*ATTACK_STEER_RATIO*driving_direction

            else: 
                print("didn't see the puck")
                ori_to_last_see = get_vector_from_this_to_that(pos_me, last_see)
                turn_project = calculate_turn_angle(ori_me, ori_to_last_see)

                # if turn_project > 1e-25:
                #     action["steer"] = np.sign(np.cross(ori_to_last_see, ori_me))*turn_project*ATTACK_STEER_RATIO*driving_direction
                action["steer"] = driving_direction * last_see

                if self.rescue_count % 20 == 0:
                    action["rescue"]=True
                
                self.last_see_count += 1
                if self.last_see_count % 5 == 0:
                    self.last_see_position = self.center

                self.rescue_count += 1
                action["brake"] = 1
                action["acceleration"] = 0

        else:
            # print("guard")
            ori_to_guard_loc = get_vector_from_this_to_that(pos_me, self.guard_loc,normalize=False)
            ori_to_guard_loc_n = get_vector_from_this_to_that(pos_me, self.guard_loc)
            distance_to_guard_loc = np.linalg.norm(ori_to_guard_loc)
            protect_action = False
            if puck_location is not None:
                if (puck_location[1]<-20): 
                    protect_action = True
            if puck_location is not None:
                self.last_see_position = puck_location

            if protect_action:
                
                self.his_buffer.push(puck_location)
                ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
                ori_puck_to_goal_n = get_vector_from_this_to_that(self.goal, puck_location, normalize=True)
            
                action["acceleration"] = 1

                if len(self.his_buffer)>2: #and get_vector_from_this_to_that(_his[0],_his[1],normalize=False)[1]<-1:
                    his = self.his_buffer.get_history(2)
                    speed = get_vector_from_this_to_that(his[0], his[1], normalize=False)
                    pos_hit_loc = puck_location + 1.2 * speed
                else:
                    pos_hit_loc = puck_location + 0.5 * ori_puck_to_goal_n
                
                ori_to_puck_n = get_vector_from_this_to_that(pos_me, pos_hit_loc)
                turn_project = calculate_turn_angle(ori_me, ori_to_puck_n)
                if turn_project > 1e-25:
                    action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me)) * DEFEND_STEER_RATIO * driving_direction
            else:
                if (distance_to_guard_loc < 2): #Goalie isnt at goal keeper location
                    if kart_vel < 0:
                        action["acceleration"] = abs(kart_vel/10)
                    if puck_location is not None: # We can see the puck
                        self.his_buffer.push(puck_location)
                    else: 
                        puck_location = self.center
                    
                    ori_to_puck = get_vector_from_this_to_that(pos_me, puck_location,normalize=False)
                    ori_to_puck_n = get_vector_from_this_to_that(pos_me, puck_location)
                    
                    turn_project = calculate_turn_angle(ori_me, ori_to_puck_n)
                    
                    if turn_project > .0005:
                        if np.dot(ori_to_guard_loc , ori_me)<0:
                            action["brake"] = 1.
                            action["acceleration"] = 0.0
                        else:
                            action["acceleration"] = (abs(kart_vel)+.2)/4.5
                        action["steer"] = np.sign(np.cross(ori_to_puck_n, ori_me))*driving_direction*DEFEND_STEER_RATIO
                else:
                    if np.dot(ori_to_guard_loc, ori_me) > 0: 
                        action["acceleration"] = 0.2
                    else:
                        action["brake"] = 1.
                        action["acceleration"] = 0.0
                    
                    turn_project = calculate_turn_angle(ori_me, ori_to_guard_loc_n)
                    if turn_project > 1e-25:
                        action["steer"] = np.sign(np.cross(ori_me, ori_to_guard_loc_n))*DEFEND_STEER_RATIO*driving_direction            

        if action["steer"] > 0.5: action["drift"] = True 

        return action