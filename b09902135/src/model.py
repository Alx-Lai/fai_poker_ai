import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias.data)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attr_linear = nn.Sequential( # 1*16 -> 1*128
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.card_conv = nn.Sequential( # 1*4*16*16 -> 128*1*1
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.concat_linear = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.Softmax(dim=0)
        )
    
    def forward(self, attr, card):
        out1 = self.attr_linear(attr) # 1*128
        out2 = self.card_conv(card) # 128*1*1
        out1 = torch.squeeze(out1)
        out2 = torch.squeeze(out2)
        if len(out1.shape) == 2:
            concat = torch.cat((out1, out2), dim=1)
        else:
            concat = torch.cat((out1, out2))
        out = self.concat_linear(concat)

        return out


class DQN():

    def __init__(self, mem_size = 5000, learning_rate=1e-4, replace_frequency=50, batch_size=128, gamma=0.9, device=1):
        self.target = Model()
        self.eval = Model()
        self.target.apply(weight_init)
        self.eval.apply(weight_init)
        self.learning_rate = learning_rate
        self.replace_frequency = replace_frequency
        self.learn_counter = 0
        self.memory_counter = 0
        self.mem_size = mem_size
        self.tmp_memory = [[torch.zeros(16), torch.zeros((4,16,16)), torch.zeros(16), torch.zeros((4,16,16)), 0, 0] for _ in range(self.mem_size)]
        self.tmp_memory_counter = 0
        self.memory = [[torch.zeros(16), torch.zeros((4,16,16)), torch.zeros(16), torch.zeros((4,16,16)), 0, 0] for _ in range(self.mem_size)]
        self.veryBIG = 1000
        self.td_err = [self.veryBIG]*self.mem_size
        self.optimizer = AdamW(self.eval.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()
        self.start_epsilon = 1
        self.epsilon = self.start_epsilon
        self.end_epsilon = 0.15
        self.annealing_step = 20000
        self.N_actions = 6
        self.batch_size = batch_size
        self.alpha = 0.6 # td err
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
            self.eval = self.eval.to(self.device)
            attr = attr.to(self.device)
            cards = cards.to(self.device)
            predict = self.eval.forward(attr, cards)
            predict = predict.to('cpu')
            return torch.argmax(predict)
    
    def push_state(self, attr, cards, reward, action):
        self.tmp_memory[self.tmp_memory_counter%self.mem_size] = [self.old_attr, self.old_cards, attr, cards, reward, action]
        self.tmp_memory_counter += 1
    
    def push_result(self, reward, action):
        self.memory[self.memory_counter%self.mem_size] = [self.old_attr, self.old_cards, self.old_attr, self.old_cards, reward, action]
        self.memory_counter += 1
        for i in range(self.tmp_memory_counter):
            self.memory[self.memory_counter%self.mem_size] = self.tmp_memory[i]
            self.memory[self.memory_counter%self.mem_size][4] = reward
            self.td_err[self.memory_counter%self.mem_size] = self.veryBIG
            self.memory_counter += 1
        self.tmp_memory_counter = 0
        self.old_attr = torch.zeros(16)
        self.old_cards = torch.zeros((4,16,16))
        if not self.learn_ok and (self.memory_counter + 1) >= self.mem_size:
            self.learn_ok = True
            print(">>> start learn <<<")
    
    def clear_state(self):
        self.old_attr = torch.zeros(16)
        self.old_cards = torch.zeros((4,16,16))

    def learn(self):
        if self.learn_counter % self.replace_frequency == 0:
            self.target.load_state_dict(self.eval.state_dict())
        self.learn_counter += 1
        s = sum(self.td_err)
        prob = [self.td_err[i]/s for i in range(self.mem_size)]
        # print(type(prob), type(self.mem_size), type(self.batch_size))

        idxs = np.random.choice(self.mem_size, size=self.batch_size, replace=False, p=prob)
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
        
        self.gamma = 1 - 0.98 * self.gamma # https://zhuanlan.zhihu.com/p/37685044
        loss = self.loss(q_eval, q_target)
        for i in range(self.batch_size):
            self.td_err[idxs[i]] = q_target[i] - q_eval[i]
            self.td_err[idxs[i]] = abs(self.td_err[idxs[i]][0].to('cpu').item())
            self.td_err[idxs[i]] = (self.td_err[idxs[i]]+0.1)**self.alpha

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def update_random(self):
        if self.epsilon > self.end_epsilon and self.learn_ok:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / self.annealing_step