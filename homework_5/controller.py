import pystk
import math



def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    brake_angle = 110 * math.pi / 180
    drift_angle = 120 * math.pi / 180
    steer_angle = 165 * math.pi / 180
    target_velocity = 18
    max_velocity = 35
    
    angle = math.atan2(aim_point[0], aim_point[1])
    brake = abs(angle) <  brake_angle and current_vel >=  target_velocity
    steer = abs(angle) < steer_angle
    drift = abs(angle) < drift_angle
    nitro = current_vel <= max_velocity and drift == False

    action.brake = brake
    angle = angle if steer else 0
    scale = 0.65
    action.steer = angle * scale
    action.drift = drift and not brake
    action.acceleration = 1 if current_vel < max_velocity and not action.brake else 0
    action.nitro = nitro
    
    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
