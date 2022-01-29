import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from game import WordleGame
from solver import words, filter_wordlist, elimination_solver, update_mask
from bisect import bisect_left
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

class WordleNet(nn.Module):
    """End-to-end network to solve Wordle games."""
    
    def __init__(self, hidden_dim=4096):
        input_shape = (1, 2315, 5)
        super(WordleNet, self).__init__()
        self.word_list = sorted(words)
        self.flatten = nn.Flatten()
        self._input = nn.Linear(input_shape[1] * input_shape[2], hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden4 = nn.Linear(hidden_dim, input_shape[1])
        self.output = nn.Softmax()
    
    def forward(self, x):
        x = F.relu(self._input(self.flatten(x)))
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        return self.output(x)


def pretrain(net, device='cuda', epochs=100, sample_games=500):
    """Pretrain the network on a sample of games."""
    # generate and run 100 games using elimination solver
    games = []
    print('Generating games...')
    for i in tqdm(range(sample_games)):
        w = WordleGame()
        games.append(w)
        elimination_solver(w)
    # iterate through the games and learn
    assert next(net.parameters()).device.type == torch.device(device).type
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_hist = []
    print('Pretraining...')
    for i in tqdm(range(epochs)):
        game = random.choice(games)
        running_X = torch.zeros(1, 2315, 5).to(device)
        target_y = torch.zeros(1, 2315).to(device)
        # set target word as 1 in the mask
        target_y[0, bisect_left(net.word_list, game.target_word)] = 1
        
        # pick random step to train from 
        step = random.randint(0, game.curr_step - 1)
        running_X[0, bisect_left(net.word_list, game.word_list[step]), :] = torch.Tensor(game.accuracy_list[step])
        move = net(running_X)
        # calculate loss
        loss = loss_fn(move, torch.max(target_y, 1)[1])
        loss_hist.append(loss.item())
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Game {i}; Loss: {loss}")
    
    plt.plot(loss_hist)
    plt.savefig("./test.png")

# TODO: move masking into the softmax layer (using weight parameter)
def train(net, device='cuda', epochs=500):
    """Train the network."""
    torch.autograd.set_detect_anomaly(True)

    # 2315 will be organized in alphabetical order to make search more efficient
    assert next(net.parameters()).device.type == torch.device(device).type
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    score_history = []
    score_elim_history = []
    for epoch in tqdm(range(epochs)):
        running_X = torch.zeros(1, 2315, 5).to(device)
        target_y = torch.zeros(1, 2315).to(device)
        wg = WordleGame()
        # try elimination solver as a baseline, then reset the same game
        score_elim_history.append(elimination_solver(wg))
        wg.reset()
        # wg.verbose = 2
        # set target word as 1 in the mask
        target_y[0, bisect_left(net.word_list, wg.target_word)] = 1
        # masking objects
        mask = ["."] * 5 # mask for each letter
        poss = set() # required letters
        filter_mask = torch.zeros(1, 2315).to(device)
        # solve game, backpropping on each turn
        while not wg.solved and wg.curr_step < 15:
            # PRE-BACKPROP: --------------------------------------------------
            optimizer.zero_grad()
            move = net(running_X)
            # subtract filter_mask to target_y
            masked_y = target_y - filter_mask

            # get index of the highest probability after subtracting mask
            move_idx = torch.argmax(move - filter_mask).item()
            # print(torch.max(move - filter_mask).item(), torch.max(move).item())
            # get the word at that index
            move_word = net.word_list[move_idx]
            # make the turn
            result = wg.turn(move_word)
            # if wg.curr_step == 15:
            #     print("Limit reached.")

            # BACKPROP: -------------------------------------------------
            # calculate loss
            # TODO: find a decent loss function
            loss = loss_fn(move, masked_y)
            # backprop
            loss.backward()

            # POST-BACKPROP: --------------------------------------------------
            # update running_X
            running_X[0, move_idx, :] = torch.tensor(result)
            # update filter_mask
            mask, poss = update_mask(result, move_word, mask, poss)
            still_valid_words = set(filter_wordlist(mask, poss, net.word_list))
            for i in range(2315):
                if net.word_list[i] not in still_valid_words:
                    filter_mask[0, i] = 1  # mask this word if no longer valid
            # add guessed word to filter_mask
            filter_mask[0, move_idx] = 1

        optimizer.step()

        # print("Epoch {}: {}".format(epoch, wg.score))
        score_history.append(wg.score)

    plt.plot(score_elim_history, c='red', alpha=0.5, label='Elimination Solver')
    plt.plot(score_history, c='blue', alpha=0.8, label='Network')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title("Training Results")
    plt.savefig("./test.png")

    print(f"Network beats Elimination Solver {round((np.sum(np.array(score_history) > np.array(score_elim_history)) / epochs) * 100, 3)}% of the time.")

if __name__ == '__main__':
    net = WordleNet().cuda()
    # pretrain(net)
    train(net, epochs=1000)
    
