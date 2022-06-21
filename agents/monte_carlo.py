from game.players import BasePokerPlayer
from agents.card_utils import CardUtil
from game.engine.card import Card
import random
from copy import deepcopy


class Player(BasePokerPlayer):
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def __init__(self):
        super().__init__()
        self.card_util = CardUtil()
        self.run_time = 10000
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        decision = {
            'fold':(valid_actions[0]['action'], valid_actions[0]['amount']),
            'call':(valid_actions[1]['action'], valid_actions[1]['amount']),
            'raise_small':(valid_actions[2]['action'], valid_actions[2]['amount']['min']),
            'raise_medium':(valid_actions[2]['action'], valid_actions[2]['amount']['max']/2),
            'raise_big':(valid_actions[2]['action'], valid_actions[2]['amount']['max']),
            }


        cards = [Card.from_str(card) for card in hole_card]
        community_card = [Card.from_str(card) for card in round_state['community_card']]
        deck = [i+1 for i in range(52)]

        for card in community_card:
            deck.remove(card.to_id())
        for card in cards:
            deck.remove(card.to_id())
        
        win_time = 0
        for _ in range(self.run_time):
            deck2 = deepcopy(deck)
            mycards = []
            othercards = []
            new_card = random.sample(deck2 ,5-len(community_card))
            for i in range(len(new_card)):
                deck2.remove(new_card[i])
                new_card[i] = Card.from_id(new_card[i])
            
            for i in range(len(new_card)):
                mycards.append(new_card[i])
                othercards.append(new_card[i])
            for i in range(len(community_card)):
                mycards.append(community_card[i])
                othercards.append(community_card[i])

            otherhand = random.sample(deck2, 2)
            othercards.append(Card.from_id(otherhand[0]))
            othercards.append(Card.from_id(otherhand[1]))
            mycards.append(Card.from_str(hole_card[0]))
            mycards.append(Card.from_str(hole_card[1]))

            win_time += self.card_util.win(mycards, othercards)
        
        win_rate = win_time / self.run_time
        # print(f'>>> winrate = {win_rate} <<< ')

        if win_rate > 0.85:
            return decision['raise_medium']
        elif win_rate > 0.75:
            return decision['raise_small']
        elif decision['call'][1] > 100:
            if win_rate > 0.7:
                return decision['call']
            return decision['fold']
        elif win_rate > 0.5:
            return decision['call']
        elif win_rate < 0.3:
            return decision['fold']
        elif decision['call'][1] == 0 or \
            (round_state['street'] == 'preflop' and decision['call'][1] <= self.game_info['rule']['small_blind_amount']):
            return decision['call']
        else:
            return decision['fold']

    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return Player()
