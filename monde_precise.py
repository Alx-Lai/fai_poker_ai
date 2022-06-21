from game.engine.card import Card
from agents.card_utils import CardUtil
import pickle
import matplotlib.pyplot as plt
import random


card_util = CardUtil()
dataset_size = 10
monte_carlo_times = [100, 500, 1000, 2000]


def variance(x):
    mean = sum(x)/len(x)
    div = 0
    for i in range(len(x)):
        div += (x[i]-mean)**2
    return div / len(x)

win_rates = []
var = []
for monte_carlo_time in monte_carlo_times:
    for epoch in range(dataset_size):
        deck = [i+1 for i in range(52)]
        my_card_num = random.sample(deck, 2)
        for card in my_card_num:
            deck.remove(card)
        my_card = [Card.from_id(card) for card in my_card_num]
        community_card_count = 0
        community_card_num = random.sample(deck, community_card_count)
        community_card = [Card.from_id(card) for card in community_card_num]
        win_rate = card_util.estimate_win_rate(my_card, community_card, time=monte_carlo_time)
        win_rates.append(win_rate)

    var.append(variance(win_rates))


plt.plot(monte_carlo_times, var, 'o-')
plt.title('monte carol var-time')
plt.savefig('monte_carol_var-time.png')
