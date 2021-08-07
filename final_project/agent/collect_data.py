
from pathlib import Path
from PIL import Image
import argparse
import pystk
from time import time
import numpy as np
import _pickle as pickle
import random
import uuid
import os
import sys
from . import gui


def to_image(x, proj, view):
    W, H = 400, 300
    p = proj @ view @ np.array(list(x) + [1])
    return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])


def to_numpy(location):
    return np.float32([location[0], location[1], location[2]])


def action_dict(action):
    return {k: getattr(action, k) for k in ['acceleration', 'brake', 'steer', 'fire', 'drift']}


if __name__ == "__main__":
    # create uuid for file names
    uid = str(uuid.uuid1())
    soccer_tracks = {"soccer_field", "icy_soccer_field"}
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track', default='icy_soccer_field')
    parser.add_argument('-k', '--kart', default='')
    parser.add_argument('--team', type=int, default=0, choices=[0, 1])
    parser.add_argument('-s', '--step_size', type=float)
    parser.add_argument('-v', '--visualization', type=str, choices=list(gui.VT.__members__), nargs='+',
                        default=['IMAGE'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_dir', type=Path, default='dense_data/valid/')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('-n', '--steps', type=int, default=10000)
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        # create dirs
        if not os.path.exists(args.save_dir / '0'):
            os.makedirs(args.save_dir / '0')
        if not os.path.exists(args.save_dir / '1'):
            os.makedirs(args.save_dir / '1')
        if not os.path.exists(args.save_dir / '2'):
            os.makedirs(args.save_dir / '2')
        if not os.path.exists(args.save_dir / '3'):
            os.makedirs(args.save_dir / '3')

    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    pystk.init(config)

    # high possibility of wilber on opposite team
    possible_karts = ['tux', 'gnu', 'nolok', 'sara', 'adiumy', 'konqi', 'kiki', 'beastie',
                      'amanda', 'emule', 'suzanne', 'gavroche', 'hexley', 'xue', 'pidgin', 'puffy', 'wilber', 'wilber', 'wilber', 'wilber', 'wilber', 'wilber', 'wilber', 'wilber', 'wilber']

    config = pystk.RaceConfig()
    config.num_kart = 4

    config.difficulty = 2

    num_player = 4
    config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL
    for i in range(3):
        config.players.append(
            pystk.PlayerConfig(random.choice(possible_karts), pystk.PlayerConfig.Controller.AI_CONTROL, (args.team + i + 1) % 2))

    config.players[0].team = args.team

    for p in config.players:
        if((p.team) % 2 == 1):
            p.kart = random.choice(possible_karts)
        else:
            p.kart = 'wilber'

    if args.track is not None:
        config.track = args.track
        if args.track in soccer_tracks:
            config.mode = config.RaceMode.SOCCER
    if args.step_size is not None:
        config.step_size = args.step_size

    race = pystk.Race(config)
    race.start()

    if (args.display):
        uis = [gui.UI([gui.VT[x] for x in args.visualization])
               for i in range(num_player)]
    save_depth = "DEPTH" in args.visualization
    save_labels = "SEMANTIC" in args.visualization or "INSTANCE" in args.visualization

    state = pystk.WorldState()
    t0 = time()
    n = 0
    while (n < args.steps) and ((not args.display) or all(ui.visible for ui in uis)):
        if (not args.display) or (not all(ui.pause for ui in uis)):

            action = pystk.Action()
            action.acceleration = 1.0
            action.steer = np.random.uniform(-1, 1)

            race.step(action)
            state.update()
            if args.verbose and config.mode == config.RaceMode.SOCCER:
                print('Score ', state.soccer.score)
                print('      ', state.soccer.ball)
                print('      ', state.soccer.goal_line)

        if(args.display):
            for ui, d in zip(uis, race.render_data):
                ui.show(d)

        if args.save_dir:
            pos_ball = to_numpy(state.soccer.ball.location)

            for i in 0, 1, 2, 3:
                if n >= 1000 and i == 0:
                    continue
                image = np.array(race.render_data[i].image)

                # get kart, ball positions on screen of this player
                proj = np.array(state.players[i].camera.projection).T
                view = np.array(state.players[i].camera.view).T
                local_ball = to_image(pos_ball, proj, view)
                # process images that do not contain pucks
                ##
                # bx = (local_ball[0] < 0) | (local_ball[0] > 400)
                # by = (local_ball[1] < 0) | (local_ball[1] > 300)
                # slocal_ball = np.array([0, 0])
                # if bx | by:
                #     slocal_ball[0] = -1.1
                #     slocal_ball[1] = -1.1
                # else:
                #     slocal_ball[0] = local_ball[0]/400*2-1
                #     slocal_ball[1] = local_ball[1]/300*2-1
                ##
                # save locations
                np.savez(args.save_dir / str(i) /
                         ('_%06d_pos_ball' % n), local_ball)
                # np.savez(args.save_dir / str(i) /
                #          ('_%06d_spos_ball' % n), local_ball)

                (args.save_dir / str(i) /
                 ('_%06d_pos_ball.txt' % n)).write_text(str(local_ball))
                # (args.save_dir / str(i) /
                #  ('_%06d_spos_ball.txt' % n)).write_text(str(slocal_ball))

                # save images
                Image.fromarray(image).save(
                    args.save_dir / str(i) / ('_%06d_img.png' % n))

        # Make sure we play in real time
        n += 1

        sys.stdout.write("frame " + str(n))
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.flush()

        if(args.display):
            delta_d = n * config.step_size - (time() - t0)
            if delta_d > 0:
                ui.sleep(delta_d)

    race.stop()
    del race
    # pystk.clean()
