import logging
import pickle
import random
import itertools

logger = logging.getLogger(__name__)

from sample_players import BasePlayer

class BaselinePlayer(BasePlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move using iterative deepening
        #print(self.__class__.__name__)

        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            d = 1
            while(True):
                #print(d)
                self.queue.put(self.alpha_beta_search(state, depth=d))
                d += 1

    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                else:
                    beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                else:
                    alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("+inf")
        best_score = float("-inf")
        best_move = None

        for a in state.actions():
            v = min_value(state.result(a), alpha, beta, depth - 1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

class HeuristicPlayer1(BaselinePlayer):
    def score(self, state):
        depth = state.ply_count
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties)

class HeuristicPlayer2(BaselinePlayer):
    def score(self, state):
        depth = state.ply_count
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - 3*len(opp_liberties)

class HeuristicPlayer3(BaselinePlayer):
    def score(self, state):
        depth = state.ply_count
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - depth*len(opp_liberties)

class HeuristicPlayer4(BaselinePlayer):
    def score(self, state):
        depth = state.ply_count
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - max(1,4-depth)*len(opp_liberties)

class CustomPlayer(HeuristicPlayer1):
    # Created this class just to align with udacity submit expectations
    # Switch base class to change customplayer behavior
    # name is just a dummy variable to make sure inheritance works
    name = "Custom Player"
