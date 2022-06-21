import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.agent import setup_ai as myai
from agents.monte_carlo import setup_ai as myai2
from agents.wise_agent import setup_ai as myai3

from baseline0 import setup_ai as baseline0_ai
from baseline1 import setup_ai as baseline1_ai
from baseline2 import setup_ai as baseline2_ai
from baseline3 import setup_ai as baseline3_ai


config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="baseline", algorithm=baseline1_ai())
config.register_player(name="me", algorithm=myai())

## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
win_count = 0
run_time = 10
for i in range(run_time):
    game_result = start_poker(config, verbose=1)
    if game_result['players'][0]['stack'] < game_result['players'][1]['stack']:
        win_count += 1
    print(json.dumps(game_result, indent=4))
print(f'{win_count} out of {run_time}')