from game.engine.card import Card
from copy import deepcopy
import random
import torch

class CardUtil:
    
    @staticmethod
    def is_four(cards:list) -> int: 
        if len(cards) < 4: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        for i in range(len(count)-1, -1, -1):
            if count[i] >= 4: return i
        return -1
    
    @staticmethod
    def is_full_house(cards:list) -> int:
        if len(cards) < 5: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        count3 = -1
        count2 = 0
        for i in range(len(count)-1, -1, -1):
            if count[i] >= 3 and count3 != -1: count3 = i
            if count[i] >= 2: count2 += 1
        if count3 != -1 and count2 >= 2: return count3
        return -1
    
    @staticmethod
    def is_flush(cards:list) -> int:
        if len(cards) < 5: return -1
        count = [0]*0x1f
        for card in cards:
            count[card.suit] += 1
        for i in (2,4,8,16):
            if count[i] >= 5:
                l = []
                for card in cards:
                    if card.suit == i:
                        l.append(card.rank)
                l.sort(reverse=True)
                return l[1] + (l[0]<<4)
        return -1
    
    @staticmethod
    def is_straight(cards:list) -> int:
        if len(cards) < 5: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        count[1] = count[14]
        for i in range(10, 0, -1):
            ok = True
            for j in range(5):
                if count[i+j] == 0:
                    ok = False
                    break
            if ok: return i
        return -1
    
    @staticmethod
    def is_three(cards:list) -> int:
        if len(cards) < 3: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        for i in range(len(cards)-1, -1, -1):
            if count[i] >= 3: return i
        return -1

    @staticmethod
    def is_two_pair(cards:list) -> int:
        if len(cards) < 4: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        l = []
        for i in range(len(cards)):
            if count[i] >= 2:
                l.append(i)
        if len(l) < 2: return -1
        l.sort(reverse=True)
        return l[1] + (l[0]<<4)
    
    @staticmethod
    def is_pair(cards:list) -> int:
        if len(cards) < 2: return -1
        count = [0]*0xf
        for card in cards:
            count[card.rank] += 1
        for i in range(len(cards), -1, -1):
            if count[i] >= 2: return i
        return -1
    
    @staticmethod
    def is_straight_flush(cards:list) -> int:
        if len(cards) < 5: return -1
        count = [0]*0x1f
        for card in cards:
            count[card.suit] += 1
        for i in (2,4,8,16):
            if count[i] >= 5:
                l = [0]*0xf
                for card in cards:
                    if card.suit == i:
                        l[card.rank] += 1
                l[1] = l[14]
                for j in range(10, 0, -1):
                    ok = True
                    for k in range(5):
                        if l[j+k] == 0:
                            ok = False
                            break
                    if ok: return j
        return -1

    @staticmethod
    def is_royal_flush(cards:list) -> int:
        if len(cards) < 5: return -1
        count = [0]*0x1f
        for card in cards:
            count[card.suit] += 1
        for i in (2,4,8,16):
            if count[i] >= 5:
                l = [0]*0xf
                for card in cards:
                    if card.suit == i:
                        l[card.rank] += 1
                for j in range(5):
                    if l[10+j] == 0:
                        return -1
                    return i
        return -1
    
    @staticmethod
    def is_high(cards:list) -> int:
        if len(cards) < 5: return -1
        l = []
        for card in cards:
            l.append(card.rank)
        l.sort(reverse=True)
        return l[1] + (l[0]<<4)

    @classmethod
    def evaluate_score(cls, cards:list) -> int:
        if (ret := cls.is_straight_flush(cards)) != -1: return 1<<16 | ret
        if (ret := cls.is_four(cards)) != -1: return 1<<15 | ret
        if (ret := cls.is_full_house(cards)) != -1: return 1<<14 | ret
        if (ret := cls.is_flush(cards)) != -1: return 1<<13 | ret
        if (ret := cls.is_straight(cards)) != -1: return 1<<12 | ret
        if (ret := cls.is_three(cards)) != -1: return 1<<11 | ret
        if (ret := cls.is_two_pair(cards)) != -1: return 1<<10 | ret
        if (ret := cls.is_pair(cards)) != -1: return 1<<9 | ret
        return cls.is_high(cards)
    
    @classmethod
    def win(cls, cards1, cards2) -> float:
        """1 win 2 or not"""
        sc1 = cls.evaluate_score(cards1)
        sc2 = cls.evaluate_score(cards2)
        if sc1 > sc2:
            return 1
        elif sc1 == sc2:
            return 0.5
        else:
            return 0

    def estimate_win_rate(self, hole_cards:list, community_cards:list = [], time:int = 1000) -> float:
        deck = [i+1 for i in range(52)]
        for card in hole_cards:
            deck.remove(card.to_id())
        for card in community_cards:
            deck.remove(card.to_id())
        win_time = 0
        for _ in range(time):
            deck2 = deepcopy(deck)
            mycards = []
            othercards = []
            new_card = random.sample(deck2 ,5-len(community_cards))
            for i in range(len(new_card)):
                deck2.remove(new_card[i])
                new_card[i] = Card.from_id(new_card[i])
            
            for i in range(len(new_card)):
                mycards.append(new_card[i])
                othercards.append(new_card[i])

            for i in range(len(community_cards)):
                mycards.append(community_cards[i])
                othercards.append(community_cards[i])

            otherhand = random.sample(deck2, 2)
            othercards.append(Card.from_id(otherhand[0]))
            othercards.append(Card.from_id(otherhand[1]))
            mycards.append(hole_cards[0])
            mycards.append(hole_cards[1])

            win_time += self.win(mycards, othercards)
        
        win_rate = win_time / time
        return win_rate

    @staticmethod
    def form_matrix(hole_cards, community_cards):
        ret = torch.zeros([8,16,16])
        for i in range(len(hole_cards)):
            ret[i][hole_cards[i].suit-1][hole_cards[i].rank] = 1
        for i in range(len(community_cards)):
            ret[2+i][community_cards[i].suit-1][community_cards[i].rank] = 1
        for i in range(7):
            ret[7] += ret[i]
        return ret

    @staticmethod
    def evaluate_hand(cards:list) -> int:
        pair_point = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0,11, 1, 2, 2, 1, 0, 0, 1, 2, 2, 3, 4, 7],
            [0, 0, 7,11, 3, 4, 3, 1, 1, 1, 2, 2, 3, 5, 7],
            [0, 0, 7, 8,11, 5, 4, 3, 2, 1, 2, 3, 4, 5, 8],
            [0, 0, 8, 9,10,12, 6, 5, 4, 3, 2, 3, 4, 5, 8],
            [0, 0, 6, 8,10,11,13, 6, 5, 4, 4, 3, 4, 6, 7],
            [0, 0, 6, 7, 9,10,11,13, 7, 6, 6, 5, 5, 6, 8],
            [0, 0, 6, 6, 8, 9,11,12,15, 8, 8, 7, 7, 7, 9],
            [0, 0, 7, 7, 7, 8,10,12,13,16,10, 9, 9, 9,10],
            [0, 0, 8, 8, 8, 8,10,11,13,15,19,13,12,13,13],
            [0, 0, 8, 9, 9, 9, 9,11,13,15,18,22,14,14,14],
            [0, 0, 9,10,10,10,11,11,13,15,18,19,27,16,16],
            [0, 0,11,11,11,12,12,13,13,15,18,20,21,35,19],
            [0, 0,13,14,14,15,14,14,15,16,19,20,22,24,40],
        ]
        ma = max(cards[0].rank, cards[1].rank)
        mi = min(cards[0].rank, cards[1].rank)
        if cards[0].suit == cards[1].suit:
            return pair_point[ma][mi]
        return pair_point[mi][ma]
    
    @staticmethod
    def from_str(card:str)-> Card:
        SUIT_MAP = {"C":2, "D":4, "H":8, "S":16}
        RANK_MAP = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
        return Card(SUIT_MAP[card[0]], RANK_MAP[card[1]])

    @staticmethod
    def from_id(card_id:str)->Card:
        suit, rank = 2, card_id
        while rank > 13:
            suit <<= 1
            rank -= 13
        return Card(suit, rank)
        
