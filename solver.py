import random
import re
from game import WordleGame
from functools import partial
from copy import deepcopy
import numpy as np
from tqdm import tqdm

words = open("words.txt").read().splitlines()


def random_solver(game):
    """Solves a game by randomly guessing words."""
    while not game.solved:
        game.turn(random.choice(words))
    return game.score


def _check_valid_word(word, mask, poss):
    """Check if word is valid given mask and possible letters."""
    r = re.compile("".join(mask))
    matches_mask = bool(r.match(word))
    matches_poss = len(poss - set(word)) == 0
    return matches_mask and matches_poss


def filter_wordlist(mask, poss, wordlist):
    """Filter wordlist based on mask and possible letters."""
    r = re.compile("".join(mask))
    return list(filter(partial(_check_valid_word, mask=mask, poss=poss), wordlist))


def update_mask(res, guess, mask, poss):
    """Get mask for word based on turn result."""
    for i, x in enumerate(res):
        if x == 1:
            mask[i] = guess[i]
        elif (
            x == 0.5
        ):  # yellow square: remove from guess position and filter for the others
            poss.add(guess[i])
            if mask[i] == ".":
                mask[i] = "[^" + guess[i] + "]"
            elif mask[i].startswith("["):
                mask[i] = mask[i][:2] + guess[i] + mask[i][2:]
        else:  # back square: remove from all positions
            for j in range(5):
                if mask[j] == ".":
                    mask[j] = "[^" + guess[i] + "]"
                elif mask[j].startswith("["):
                    mask[j] = mask[j][:2] + guess[i] + mask[j][2:]
    return mask, poss


def elimination_solver(game):
    """Solves a game by successively eliminating choices from word list."""
    poss_words = deepcopy(words)
    mask = ["."] * 5  # mask for each letter
    poss = set()  # required letters
    while not game.solved:
        r = re.compile("".join(mask))
        ftr = filter_wordlist(mask, poss, poss_words)
        guess = random.choice(ftr)
        poss_words.remove(guess)
        res = game.turn(guess)
        mask, poss = update_mask(res, guess, mask, poss)

    return game.score


def rt_solver(game):
    """Solves a game using the READY > THUMP method."""
    poss_words = deepcopy(words)
    mask = ["."] * 5  # mask for each letter
    poss = set()  # required letters
    hardcode = ["ready", "thump"]
    while not game.solved:
        if game.curr_step < len(hardcode):
            guess = hardcode[game.curr_step]
        else:
            r = re.compile("".join(mask))
            ftr = filter_wordlist(mask, poss, poss_words)
            guess = random.choice(ftr)
        poss_words.remove(guess)
        res = game.turn(guess)
        mask, poss = update_mask(res, guess, mask, poss)

    return game.score


def build_tau(words):
    """Build tau for MaxInfo solver."""
    N = len(words)
    pi = np.zeros(shape=(N, N, 5))
    for j in range(N):
        w = WordleGame(word=words[j])
        for i in range(N):
            if words[i] == words[j]:  # we're testing the target word
                pi[i, j] = np.ones(5)
            else:
                pi[i, j] = w.turn(words[i])
    return pi.sum(axis=2)


def update_word_mask(word_mask, letter_mask, poss):
    """Update word mask based on guess."""
    for i, w in enumerate(words):
        if not _check_valid_word(w, letter_mask, poss):
            word_mask[:, i] = np.zeros(word_mask.shape[0])

    return word_mask


def maximum_information_solver(game, tau):
    """Solve game using maximum information strategy."""
    N = len(words)
    word_mask = np.full_like(tau, 1)  # 1 means word is still valid, 0 means it's not
    letter_mask = ["."] * 5  # mask for each letter
    poss = set()  # required letters
    while not game.solved:
        # reduce tau by elementwise multiplying by word_mask
        tau = np.multiply(tau, word_mask)
        # get index with highest sum (we could use average but that's just divide by N)
        word_idx = np.argmax(tau.sum(axis=0))
        # make move using this word
        guess = words[word_idx]
        result = game.turn(guess)
        # update masks based on result
        letter_mask, poss = update_mask(result, guess, letter_mask, poss)
        word_mask = update_word_mask(word_mask, letter_mask, poss)
    return game.score

