import random


class WordleGame:
    def __init__(self, word=None, verbose=0):
        self.solved = False
        self.curr_step = 0
        self.score = 10
        self.verbose = verbose  # 0: silent, 1: print final score, 2: print score and each turn result
        words = open("words.txt").read().splitlines()
        self.target_word = random.choice(words) if word is None else word
        self.word_list = []
        self.accuracy_list = []

    def _eval_letter(self, ltr, pos):
        """Return square for letter."""
        if self.target_word[pos] == ltr:
            return 1
        elif ltr in set(self.target_word):
            return 0.5
        else:
            return 0

    def turn(self, guess):
        if len(guess) != 5:
            raise ValueError("Guess must be 5 letters long")
        # if self.solved:
        #     raise ValueError("Game is already solved")

        self.curr_step += 1
        self.word_list.append(guess)

        # get correct guesses
        result = [self._eval_letter(ltr, pos) for pos, ltr in enumerate(guess)]
        self.score -= 5 - sum(result)
        if self.verbose == 2:
            print(
                guess
                + " => "
                + " ".join(
                    ["🟩" if i == 1 else "🟨" if i == 0.5 else "⬛" for i in result]
                )
            )

        if guess == self.target_word:
            # compute final score
            self.solved = True
            if self.verbose >= 1:
                print(f"Wordle solved in {self.curr_step} steps. Score: {self.score}")

        self.accuracy_list.append(result)
        return result

    def reset(self):
        """Reset game."""
        self.solved = False
        self.curr_step = 0
        self.score = 10
        self.word_list = []
        self.accuracy_list = []


def test_scoring():
    """Test scoring is monotonically decreasing on turns."""
    scores = []
    for i in range(7):
        w = WordleGame("apple")
        for j in range(i):
            w.turn("appla")
        w.turn("apple")
        scores.append(w.score)
    return scores

