from win_model import Model
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from agents.card_utils import CardUtil
from game.engine.card import Card
import pickle

class Trainer():
    def __init__(self, args, model, optimizer, loss) -> None:
        self.args = args
        self.memory = []
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.save_counter = 0
    
    def fit(self, dataset, dataset2):
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        print('[>] Dataloader Done')
        self.model = self.model.to(self.args.device)
        self.loss = self.loss.to(self.args.device)
        for epoch in tqdm(range(self.args.epoch)):
            for i, (x, winrate) in enumerate(dataloader):
                winrate = winrate.unsqueeze(dim=1)
                winrate = winrate.to(self.args.device)
                x = x.to(self.args.device)
                out = self.model(x)
                loss = self.loss(out, winrate)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 50 == 0:
                self.eval(dataset2)
    
    def eval(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        total_loss = 0 
        with torch.no_grad():
            for i, (x, winrate) in enumerate(dataloader):
                x = x.to(self.args.device)
                out = self.model(x)
                winrate = winrate.unsqueeze(dim=1)
                winrate = winrate.to(self.args.device)
                loss = self.loss(out, winrate)
                total_loss += loss.item()
            total_loss /= len(dataloader)
            torch.save(self.model, f'win_rate_model{self.save_counter}')
            self.save_counter += 1
        print(f'{total_loss=}')

        return total_loss

    def predict(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        outs = []
        with torch.no_grad():
            for i, x in tqdm(enumerate(dataloader)):
                x = x.to(self.args.device)
                out = self.model(x)
                out = out.to('cpu')
                outs.append(out)
        return outs

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--trainset_path', type=str, default='')
parser.add_argument('--testset_path', type=str, default='')
args = parser.parse_args()

# set GPU
if torch.cuda.is_available() and args.gpu >= 0:
    args.device = torch.device(f'cuda:{args.gpu}')
    print(f'[>] Using CUDA {args.gpu}')
else:
    args.device = 'cpu'
    print('[>] Using cpu')

model = Model()
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
loss = nn.L1Loss()
trainer = Trainer(args, model, optimizer, loss)
card_util = CardUtil()

with open(args.trainset_path, 'rb') as handle:
    dataset = pickle.load(handle)

for i in range(len(dataset)):
    for j in range(len(dataset[i][0])):
        dataset[i][0][j] = Card.from_id(dataset[i][0][j])
    for j in range(len(dataset[i][1])):
        dataset[i][1][j] = Card.from_id(dataset[i][1][j])
train_dataset = []
for i in range(len(dataset)):
    cards_ = card_util.form_matrix(dataset[i][0], dataset[i][1])
    cards = torch.zeros(4,16,16) # 4*16*16
    cards[0] = cards_[0] + cards_[1]
    cards[1] = cards_[2] + cards_[3] + cards_[4]
    cards[2] = cards_[2] + cards_[3] + cards_[4] + cards_[5] + cards_[6]
    cards[3] = cards_[-1]
        
    train_dataset.append((cards, dataset[i][2]))

with open(args.testset_path, 'rb') as handle:
    dataset = pickle.load(handle)
test_dataset = []
for i in range(len(dataset)):

    cards_ = card_util.form_matrix(dataset[i][0][:2], dataset[i][0][2:])
    cards = torch.zeros(4,16,16) # 4*16*16
    cards[0] = cards_[0] + cards_[1]
    cards[1] = cards_[2] + cards_[3] + cards_[4]
    cards[2] = cards_[2] + cards_[3] + cards_[4] + cards_[5] + cards_[6]
    cards[3] = cards_[-1]
        
    test_dataset.append((cards, dataset[i][1]))

train_dataset1 = train_dataset[:90000]
train_dataset2 = train_dataset[90000:]
if args.do_train:
    trainer.fit(train_dataset1, train_dataset2)
else:
    pass


