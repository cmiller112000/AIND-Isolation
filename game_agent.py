"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def improved_score(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

def minmax_xy(moves):
    if len(moves) == 0:
        return -1, -1, -1, -1
    minx = min([(move[0]) for move in moves])
    maxx = max([(move[0]) for move in moves])
    miny = min([(move[1]) for move in moves])
    maxy = max([(move[1]) for move in moves])
    return minx, maxx, miny, maxy

def get_reachable(moves):
    minx, maxx, miny, maxy = minmax_xy(moves)

    moveset = set()
    if (minx < 0) :
        return moveset
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            moveset.add((x,y))
    return moveset

def open_area(game, player, blanks):
    own_moves=get_reachable(game.get_legal_moves(player))
    if len(own_moves) > 0:
        own_reachable = own_moves.intersection(blanks)
    else:
        own_reachable = set()
    opp_moves=get_reachable(game.get_legal_moves(game.get_opponent(player)))
    if len(opp_moves) > 0:
        opp_reachable = opp_moves.intersection(blanks)
    else:
        opp_reachable = set()
    return float(len(own_reachable) - len(opp_reachable))


def open_score(game, player):
    legalmoves = game.get_legal_moves(player)
    return float(len(legalmoves))


def manhatten_distance(p,q):
    return abs( p[0] - q[0]) + abs(p[1] - q[1])

def euclidean_distance(p,q):
    return math.sqrt( (p[0] - q[0])**2 + (p[1] - q[1])**2 )


def near_corner(game,player):
    isnc=False
    ploc=game.get_player_location(player)
    if ploc[0] < 2 or ploc[0] > (game.height - 2) or ploc[1] < 1 or ploc[1] > (game.width - 2):
        isnc=True
    return isnc

def w_corners(game,player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    moves_diff = own_moves - opp_moves
    own_ncorner=0
    if near_corner(game,player):
        own_ncorner=1
    opp_ncorner=0
    if near_corner(game,game.get_opponent(player)):
        opp_ncorner=1

    score = moves_diff - 2*own_ncorner + 4*opp_ncorner
    return float( score)

def is_edge(game,player):
    isnc=False
    ploc=game.get_player_location(player)
    if ploc[0] == 0 or ploc[0] == (game.height - 1) or ploc[1] == 0 or ploc[1] == (game.width - 1):
        isnc=True
    return isnc

def w_edges(game,player,blanks):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    moves_diff = own_moves - opp_moves
    own_edge=0
    opp_edge=0
    if is_edge(game,player):
        own_edge=1
    if is_edge(game,game.get_opponent(player)):
        opp_edge=1
    score =  0.4*moves_diff - 0.3*own_edge + 0.3*opp_edge
    return float( score)

def w_improved_score(game, player, distfunc=euclidean_distance):
    cntr = ( math.ceil(game.height/2), math.ceil(game.width/2) )
    own_dist = distfunc(game.get_player_location(player),cntr)
    opp_dist = distfunc(game.get_player_location(game.get_opponent(player)),cntr)
    dist_diff = opp_dist - own_dist
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    moves_diff = own_moves - opp_moves
    score = moves_diff + 2*dist_diff
    return float( score)

def common_moves(game,player):
    own_moves = set(game.get_legal_moves(player))
    opp_moves = set(game.get_legal_moves(game.get_opponent(player)))
    cmn_moves = own_moves.intersection(opp_moves)
    return float(len(cmn_moves))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------player
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    blanks=game.get_blank_spaces()
    pctblanks=len(blanks)/(game.height * game.width)
    if pctblanks < 0.5:
        return w_edges(game, player,blanks)
    elif pctblanks < 0.8:
        return open_area(game, player,blanks)
    else:
        return improved_score(game,player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.function_mappings = { "minimax": self.minimax, "alphabeta": self.alphabeta}

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        bestmove=(-1,-1)
        if not legal_moves or len(legal_moves) == 0:
            bestmove

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # if we are not doing iterative deeping, call the search method once for as deep a we're allowed
            # otherwise, we are doing iterative deeping so repeated call the search method starting at a depth 2 (top level, plus it's children)
            # and continue to increase the # of levels until we finish, or time runs out, and return the best score move seen until then

            if not self.iterative:
                _,bestmove = self.function_mappings[self.method](game, self.search_depth)
            else:
                d = 2
                if (len(legal_moves) == 0):
                    bestmove=(-1,-1)
                else:
                    bestscore, bestmove = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
                    while (True):
                        score,move = self.function_mappings[self.method](game, d)
                        if move == (-1,-1):
                            break
                        if score > bestscore:
                            bestscore=score
                            bestmove=move
                        d += 1

        except Timeout:
            pass

        # Return the best move from the last completed search iteration
        return bestmove

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if (depth == 0):
            return self.score(game,self), (-1,-1)
        lmoves = game.get_legal_moves()
        if not lmoves:
            return self.score(game,self), (-1,-1)
        bestscore = None
        if maximizing_player:
            for m in lmoves:
                score, move = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)
                if bestscore is None or score > bestscore:
                    bestscore = score
                    bestmove = m
        else:
            for m in lmoves:
                score, move = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)
                if bestscore is None or score < bestscore:
                    bestscore = score
                    bestmove = m

        return bestscore, bestmove

    def argsort(self,listtosort, reversed=False):
        return sorted(range(len(listtosort)),key=listtosort.__getitem__, reverse=reversed)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if (depth == 0):
            return self.score(game,self), (-1,-1)
        lmoves = game.get_legal_moves()
        if not lmoves:
            return self.score(game,self), (-1,-1)

        bestscore=None
        if maximizing_player:
            for m in lmoves:
                score, move = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)
                if bestscore is None or score > bestscore:
                    bestscore = score
                    bestmove = m
                    if bestscore >= beta:
                        return bestscore, bestmove
                    alpha = max(alpha,bestscore)
        else:
            for m in lmoves:
                score, move = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)
                if bestscore is None or score < bestscore:
                    bestscore = score
                    bestmove = m
                    if bestscore <= alpha:
                        return bestscore, bestmove
                    beta = min(beta,bestscore)

        return bestscore, bestmove
