import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.agent import setup_ai as myai
from agents.monte_carlo import setup_ai as myai2
from agents.wise_agent import setup_ai as myai3
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import random
import torch

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--is_train", action='store_true')
parser.add_argument("--fix_ai", type=int, default=-1)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--checkpoint_path", type=str, default='checkpoints')
args = parser.parse_args()

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
rand_ai = random_ai()
AIS = [
    {"name":"baseline0", "algorithm":rand_ai},
    {"name":"baseline1", "algorithm":rand_ai},
    {"name":"baseline2", "algorithm":rand_ai},
    {"name":"baseline3", "algorithm":rand_ai},
    ]
# set GPU
if torch.cuda.is_available() and args.gpu >= 0:
    args.device = torch.device(f'cuda:{args.gpu}')
    print(f'[>] Using CUDA {args.gpu}')
else:
    args.device = 'cpu'
    print('[>] Using cpu')

train_ai = myai(device=args.device, model_path=args.model_path, is_train=args.is_train, checkpoint=args.checkpoint_path)
config.register_player(name="baseline", algorithm=AIS[0]['algorithm'])
config.register_player(name="me", algorithm=train_ai)

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
run_time = args.epoch
win_count = {
    "baseline0":0,
    "baseline1":0,
    "baseline2":0,
    "baseline3":0,
    }
battle_count = {
    "baseline0":0,
    "baseline1":0,
    "baseline2":0,
    "baseline3":0,
    }
name2id = {
    "baseline0":0,
    "baseline1":1,
    "baseline2":2,
    "baseline3":3,
    }

recent_win = []
win_ai = [0]*4
battle_ai = [0]*4
sofar_win_rate = [[] for i in range(5)]

nowai_counter = 0
ai_row = [i for i in range(4)]
for rt in tqdm(range(run_time)):
    if args.fix_ai != -1:
        AI = AIS[args.fix_ai]
    else:
        AI = AIS[ai_row[nowai_counter]]
        nowai_counter += 1
        nowai_counter %= len(AIS)
        if nowai_counter == 0:
            random.shuffle(ai_row)
    config.players_info[0] = AI
    game_result = start_poker(config, verbose=1)
    battle_ai[name2id[game_result['players'][0]['name']]] += 1
    battle_count[game_result['players'][0]['name']] += 1
    if game_result['players'][0]['stack'] < game_result['players'][1]['stack']:
        win_count[game_result['players'][0]['name']] += 1
        #print(game_result['players'][0]['name'] , 'v.s.', game_result['players'][1]['name'], '[win]')
        win_ai[name2id[game_result['players'][0]['name']]] += 1
        recent_win.append(1)
    else:
        #print(game_result['players'][0]['name'], '[win]', 'v.s.', game_result['players'][1]['name'])
        recent_win.append(0)
    if len(recent_win) == 100:
        print(f'win rate:{sum(recent_win)/100}')
        recent_win = []
        for i in range(4):
            if battle_ai[i] != 0:
                sofar_win_rate[i].append(win_ai[i]/battle_ai[i])
            else:
                sofar_win_rate[i].append(0)
        sofar_win_rate[4].append(sum(win_ai)/sum(battle_ai))
        win_ai = [0]*4
        battle_ai = [0]*4
        plt.clf()
        x_axis = [i+1 for i in range(len(sofar_win_rate[0]))]
        for i in range(4):
            plt.plot(x_axis, sofar_win_rate[i], label=AIS[i]['name'])
        plt.plot(x_axis, sofar_win_rate[4], 'o-', label='overall')
        plt.legend()
        plt.savefig(f'images/win_rate.png')
    if not args.is_train:
        print(json.dumps(game_result, indent=4))

torch.save(train_ai.model, f'checkpoints/checkpoint{train_ai.save_model_counter}.pth')
for k in win_count.keys():
    print(f'{k} {win_count[k]}/{battle_count[k]}')