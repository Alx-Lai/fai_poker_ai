from game.players import BasePokerPlayer
from agents.card_utils import CardUtil
import numpy as np


class CallPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    
    def __init__(self):
        super().__init__()
        self.util = CardUtil()
        self.invest = 0

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        
        if valid_actions[2]['amount']['max'] != -1:
            decision = {
                'fold':(valid_actions[0]['action'], valid_actions[0]['amount']),
                'call':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_small':(valid_actions[2]['action'], valid_actions[2]['amount']['min']),
                'raise_medium1':(valid_actions[2]['action'], max(valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']//3, 150)),
                'raise_medium2':(valid_actions[2]['action'], max(valid_actions[2]['amount']['min'], (valid_actions[2]['amount']['max']*2)//3, 150)),
                'raise_big':(valid_actions[2]['action'], valid_actions[2]['amount']['max']),
                }
        else:
            decision = {
                'fold':(valid_actions[0]['action'], valid_actions[0]['amount']),
                'call':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_small':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_medium1':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_medium2':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_big':(valid_actions[1]['action'], valid_actions[1]['amount']),
                }
        other_stack = []
        for s in round_state["seats"]:
            if s["uuid"] == self.uuid:
                my_stack = s["stack"]
            else:
                other_stack.append(s["stack"])
        round = round_state["round_count"]
        ok_val = ((21-round)//2)*15
        hands = [CardUtil.from_str(hole_card[i]) for i in range(len(hole_card))]
        hand_val = self.util.evaluate_hand(hands)
        if (my_stack - 1000) >= ok_val:
            return decision['fold']
        if decision['raise_big'][1] == -1 and decision['call'][1] == 0:
            self.invest += decision['call'][1]
            return decision['call']
        if hand_val >= 25:
            action = np.random.choice(list(decision.keys()), p=[0,0.0,0.25,0.25,0.25,0.25])
            return decision[action]
        if hand_val >= 20:
            action = np.random.choice(list(decision.keys()), p=[0,0.1,0.225,0.225,0.225,0.225])
            return decision[action]
        if hand_val >= 15:
            action = np.random.choice(list(decision.keys()), p=[0,0.2,0.2,0.2,0.2,0.2])
            return decision[action]
        if decision['call'][1] <= 10:
            return decision['call']
        return decision['fold']

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.invest = 0
        
    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return CallPlayer()
