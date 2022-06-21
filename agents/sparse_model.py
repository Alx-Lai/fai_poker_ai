import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attr_linear = nn.Sequential( # 1*16 -> 1*256
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.card_conv = nn.Sequential( # 1*8*16*16 -> 256*1*1
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.Softmax(dim=0)
        )
    
    def forward(self, attr, card):
        out1 = self.attr_linear(attr) # 1*256
        out2 = self.card_conv(card) # 256*1*1
        out1 = torch.squeeze(out1)
        out2 = torch.squeeze(out2)
        if len(out1.shape) == 2:
            concat = torch.cat((out1, out2), dim=1)
        else:
            concat = torch.cat((out1, out2))
        out = self.concat_linear(concat)

        return out

class DQN():

    def __init__(self, mem_size = 5000, learning_rate=1e-4, replace_frequency=100, batch_size=128, gamma=0.8, device=1):
        self.target = Model()
        self.eval = Model()
        self.learning_rate = learning_rate
        self.replace_frequency = replace_frequency
        self.learn_counter = 0
        self.memory_counter = 0
        self.mem_size = mem_size
        self.memory = [[torch.zeros(16), torch.zeros((4,16,16)), torch.zeros(16), torch.zeros((4,16,16)), 0, 0] for _ in range(self.mem_size)]
        self.optimizer = AdamW(self.eval.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.start_epsilon = 1
        self.epsilon = self.start_epsilon
        self.end_epsilon = 0.15
        self.annealing_step = 20000
        self.N_actions = 6
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_ok = False
        self.device = device
        
        self.old_attr = torch.zeros(16)
        self.old_cards = torch.zeros((4,16,16))

    def pick_action(self, attr:torch.Tensor, cards:torch.Tensor):
        if self.epsilon > np.random.random():
            return np.random.randint(0, self.N_actions)
        else:
            # to device
            self.eval.to(self.device)
            attr = attr.to(self.device)
            cards = cards.to(self.device)
            predict = self.eval.forward(attr, cards)
            predict = predict.to('cpu')
            return torch.argmax(predict)
    
    def push_state(self, attr, cards, reward, action):
        self.memory[self.memory_counter%self.mem_size] = [self.old_attr, self.old_cards, attr, cards, reward, action]
        if not self.learn_ok and (self.memory_counter + 1) >= self.mem_size:
            self.learn_ok = True
            print(">>> start learn <<<")
        self.memory_counter += 1
    
    def push_result(self, reward, action):
        self.memory[self.memory_counter%self.mem_size] = [self.old_attr, self.old_cards, self.old_attr, self.old_cards, reward, action]
        self.old_attr = torch.zeros(16)
        self.old_cards = torch.zeros((4,16,16))
        if not self.learn_ok and (self.memory_counter + 1) >= self.mem_size:
            self.learn_ok = True
            print(">>> start learn <<<")
        self.memory_counter += 1
    
    def clear_state(self):
        self.old_attr = torch.zeros(16)
        self.old_cards = torch.zeros((4,16,16))

    def learn(self):
        if self.learn_counter % self.replace_frequency == 0:
            self.target.load_state_dict(self.eval.state_dict())
        self.learn_counter += 1

        idxs = np.random.choice(self.mem_size, self.batch_size)
        batch_memory = [self.memory[idx] for idx in idxs]
        batch_old_attr = torch.stack([batch_memory[i][0] for i in range(self.batch_size)])
        batch_old_cards = torch.stack([batch_memory[i][1] for i in range(self.batch_size)])
        batch_attr = torch.stack([batch_memory[i][2] for i in range(self.batch_size)])
        batch_cards = torch.stack([batch_memory[i][3] for i in range(self.batch_size)])
        batch_reward = torch.FloatTensor([batch_memory[i][4] for i in range(self.batch_size)])
        batch_action = torch.Tensor([batch_memory[i][5] for i in range(self.batch_size)])

        batch_old_attr = batch_old_attr.to(self.device)
        batch_old_cards = batch_old_cards.to(self.device)
        batch_attr = batch_attr.to(self.device)
        batch_cards = batch_cards.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_action = batch_action.to(self.device)
        self.eval = self.eval.to(self.device)
        self.target = self.target.to(self.device)

        batch_action = batch_action.type(torch.int64)
        batch_action = batch_action.unsqueeze(1)
        q_eval = self.eval(batch_attr, batch_cards).gather(1, batch_action)
        q_next = self.target(batch_old_attr, batch_old_cards).detach()
        q_target = batch_reward.unsqueeze(dim=1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def update_random(self):
        if self.epsilon > self.end_epsilon and self.learn_ok:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / self.annealing_step