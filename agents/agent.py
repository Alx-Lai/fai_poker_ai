from game.players import BasePokerPlayer
from agents.card_utils import CardUtil
from agents.model import DQN
import torch
import math


class Player(BasePokerPlayer):
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def __init__(self, device, model_path='agents/checkpoint21000.pth', is_train=False, checkpoint='checkpoints'):
        super().__init__()
        self.card_util = CardUtil()
        self.device = device
        if model_path is None:
            self.model = DQN(device=device)
        else:
            self.checkpoint = checkpoint
            checkpointt = torch.load(model_path, map_location='cpu')
            self.model = DQN(device=device)
            self.model.eval = checkpointt['eval']
            self.model.target = checkpointt['target']
            # self.model.optimizer = checkpointt['optimizer']
            self.model.memory = checkpointt['memory']
            self.model.epsilon = checkpointt['epsilon']
            self.model.learn_ok = True
            self.model.optimizer = torch.optim.AdamW(self.model.eval.parameters(), lr=self.model.learning_rate)
        self.last_action = 0
        self.save_model_counter = 0
        self.save_model_frequency = 3000
        self.is_train = is_train
        if not is_train:
            self.epsilon = 0
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        if valid_actions[2]['amount']['max'] != -1:
            decision = {
                'fold':(valid_actions[0]['action'], valid_actions[0]['amount']),
                'call':(valid_actions[1]['action'], valid_actions[1]['amount']),
                'raise_small':(valid_actions[2]['action'], valid_actions[2]['amount']['min']),
                'raise_medium1':(valid_actions[2]['action'], max(valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']//3)),
                'raise_medium2':(valid_actions[2]['action'], max(valid_actions[2]['amount']['min'], (valid_actions[2]['amount']['max']*2)//3)),
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
        decision_list = ['fold', 'call', 'raise_small', 'raise_medium1', 'raise_medium2', 'raise_big']
        other_stack = []
        for s in round_state["seats"]:
            if s["uuid"] == self.uuid:
                my_stack = s["stack"]
            else:
                other_stack.append(s["stack"])
        
        if not self.is_train:
            ok_value = 20 - round_state["round_count"] + 1
            if (my_stack - self.game_info['rule']['initial_stack']) > (15*(ok_value//2) + 10*(ok_value % 2)):
                return decision['fold']

        hole_cards = [self.card_util.from_str(card) for card in hole_card]
        community_cards = [self.card_util.from_str(card) for card in round_state['community_card']]
        self.features = torch.Tensor([
            1 if round_state["street"] == "preflop" else 0, # one hot encoding
            1 if round_state["street"] == "flop" else 0,
            1 if round_state["street"] == "turn" else 0,
            1 if round_state["street"] == "river" else 0,
            round_state["pot"]["main"]["amount"], # total amount
            my_stack, 
            self.round_init_stack - my_stack, # how many i throw this round
            self.street_init_stack - my_stack,# how many i throw this street
            self.game_info['rule']['initial_stack'],
            round_state["dealer_btn"],
            round_state["small_blind_pos"],
            round_state["big_blind_pos"],
            round_state["next_player"],
            round_state["round_count"],
            self.card_util.estimate_win_rate(hole_cards, community_cards)
        ] + other_stack)

        # update randomize
        self.model.update_random()

        self.cards_ = self.card_util.form_matrix(hole_cards, community_cards) # to 8*16*16 matrix
        self.cards = torch.zeros(4,16,16) # 4*16*16
        self.cards[0] = self.cards_[0] + self.cards_[1]
        self.cards[1] = self.cards_[2] + self.cards_[3] + self.cards_[4]
        self.cards[2] = self.cards_[2] + self.cards_[3] + self.cards_[4] + self.cards_[5] + self.cards_[6]
        self.cards[3] = self.cards_[-1]
        if self.is_train:
            predict = self.model.pick_action(self.features, self.cards)
            self.model.push_state(self.features, self.cards, 0, self.last_action)
            if self.model.learn_ok:
                self.model.learn()
                self.save_model_counter += 1
                if self.save_model_counter % self.save_model_frequency == 0:
                    print(f'save checkpoint{self.save_model_counter}.pth')
                    checkpointt = {
                        'eval':self.model.eval.cpu(),
                        'target':self.model.target.cpu(),
                        'optimizer':self.model.optimizer,
                        'memory':self.model.memory,
                        'epsilon':self.model.epsilon
                        }
                    torch.save(checkpointt, f'{self.checkpoint}/checkpoint{self.save_model_counter}.pth')
        else:
            with torch.no_grad():
                predict = self.model.pick_action(self.features, self.cards)

        self.last_action = predict
        self.declared = True
        # print(decision[decision_list[predict]])
        return decision[decision_list[predict]]


        
    def receive_game_start_message(self, game_info):
        self.game_info = game_info

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.declared = False
        for s in seats:
            if s["uuid"] == self.uuid:
                self.round_init_stack = s["stack"]
                break

    def receive_street_start_message(self, street, round_state):
        for s in round_state["seats"]:
            if s["uuid"] == self.uuid:
                self.street_init_stack = s["stack"]
                break

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if not self.declared:
            self.model.clear_state()
            return
        for s in round_state["seats"]:
            if s["uuid"] == self.uuid:
                my_stack = s["stack"]
                break
        earn = my_stack - self.round_init_stack
        # if earn >= 0:
        #     earn = math.log(1+earn)
        # else:
        #     earn = -math.log(1-earn)
        earn = earn / (self.game_info['rule']['small_blind_amount']*2)
        self.model.push_result(earn, self.last_action)


def setup_ai(device='cpu', model_path='agents/checkpoint21000.pth', is_train=False, checkpoint='checkpoints'):
    return Player(device=device, model_path=model_path, is_train=is_train, checkpoint=checkpoint)
