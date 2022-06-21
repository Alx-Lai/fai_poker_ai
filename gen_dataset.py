from agents.card_utils import CardUtil
import pickle
import random
from tqdm import tqdm

card_util = CardUtil()
dataset_size = 100000
monte_carlo_time = 10000

dataset = []
for epoch in tqdm(range(dataset_size)):
    deck = [i+1 for i in range(52)]
    my_card_num = random.sample(deck, 2)
    for card in my_card_num:
        deck.remove(card)
    my_card = [card_util.from_id(card) for card in my_card_num]
    community_card_count = random.choice([0,3,4,5])
    community_card_num = random.sample(deck, community_card_count)
    community_card = [card_util.from_id(card) for card in community_card_num]
    win_rate = card_util.estimate_win_rate(my_card, community_card, time=monte_carlo_time)
    
    dataset.append([my_card_num, community_card_num, win_rate])
with open('dataset.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
