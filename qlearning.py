import matplotlib.pyplot as plt
import numpy as np
import random

from nim_env import NimEnv, OptimalPlayer

# Helper functions to measure performance

def measure_performance(agent, other_epsilon):
    players = [agent, OptimalPlayer(other_epsilon)]
    N = 500
    Nwin = 0
    Nloss = 0
    env = NimEnv()
    for i in range(N):
        env.reset()
        env.current_player = i % 2

        while not env.end:
            try:
                env.step(players[env.current_player].act(env.heaps))
            except AssertionError as err:
                # End the game due to invalid move!
                env.end = True
                env.winner = env.current_player

        if env.winner == 0:
            Nwin += 1
        else:
            Nloss += 1
    return (Nwin - Nloss) / N

def Mopt(agent):
    return measure_performance(agent, 0)
def Mrand(agent):
    return measure_performance(agent, 1)

# Core Q-Learning logic
def get_possible_actions(heaps):
    '''
    Compute the list of allowed actions for a given state.

    Parameters
        ----------
        heaps : list of integers
                list of heap sizes.

    Returns
        -------
        actions : list of integers (heap number)
                  list of heap sizes (objects to remove)
    '''
    actions = []
    for i, heap_size in enumerate(heaps):
        for number in range(1, heap_size+1):
            actions.append((i+1, number))
    return actions


class LearningAgent:
    '''
    Description:
        A class to implement an epsilon-greedy learning player in Nim.

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.
    '''

    def __init__(self, epsilon):
        self.qtable = {}
        self.epsilon = epsilon

        self.last_state = None
        self.last_move = None

    def get_qvalues(self, heaps):
        '''
        Get the Q-values of all the allowed actions, for a given state.

        Parameters
            ----------
            heaps : list of integers
                    list of heap sizes.

        Returns
            ----------
            qtable : dict of action -> Q-value
        '''
        state = tuple(heaps)
        if state not in self.qtable:
            self.qtable[state] = {action: 0 for action in get_possible_actions(state)}
        return self.qtable[state]

    def _pick_best_move(self, heaps):
        '''
        Get the move with the highest Q-value for a given state, randomly breaking ties.

        Parameters
            ----------
            heaps : list of integers
                    list of heap sizes.

        Returns
            ----------
            max_act : tuple
                max_act[0] is the heap to take from (starts at 1)
                max_act[1] is the number of obj to take from heap #move[0]
        '''
        # Randomly shuffle the dictionary, in order to randomly break ties.
        qvalues = list(self.get_qvalues(heaps).items())
        random.shuffle(qvalues)
        qvalues = dict(qvalues)

        max_act = None
        for act, val in qvalues.items():
            if max_act is None or val > qvalues[max_act]:
                max_act = act
        return max_act

    def get_max_qvalue(self, heaps):
        '''
        Get the highest Q-value associated to a possible action for the given state.

        Parameters
            ----------
            heaps : list of integers
                    list of heap sizes.

        Returns
            ----------
            float (max Q-value)
        '''
        bestMove = self._pick_best_move(heaps)
        # If the game is already finished, return 0.
        return 0 if bestMove is None else self.get_qvalues(heaps)[bestMove]

    def act(self, heaps):
        '''
        Take an action, given the current state.

        Parameters
        ----------
        heaps : list of integers
                list of heap sizes.

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]

        '''
        if random.random() < self.epsilon:
            # random move
            move = random.choice(list(self.get_qvalues(heaps).keys()))
        else:
            # greedy
            move = self._pick_best_move(heaps)
        return move

# Maybe we can reuse LearningAgent later for DRL? (very unlikely) Otherwise if the hole classes have to be different,
# there is no need for the QLearningAgent class, and it can be incorporated in LearningAgent
# TODO: indeed, let's incorporate them. however we can reuse run_q_learning ;)

class QLearningAgent(LearningAgent):
    def __init__(self, epsilon, alpha=0.1, gamma=0.99):
        super().__init__(epsilon)
        self.alpha = alpha
        self.gamma = gamma

    def on_step(self, state, action, reward, new_state, debug=False):
        action = tuple(action)
        old_q_value = self.get_qvalues(state)[action]
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * self.get_max_qvalue(new_state) - old_q_value)
        self.get_qvalues(state)[action] = new_q_value

        if(debug):
            print('Last state: ', state)
            print('Last action: ', action)
            print('Old qvalue: ', old_q_value, '; new qvalue: ', new_q_value, '; reward: ', reward)

def call_on_step(agent, state, action, reward, next_state, debug):
    if hasattr(agent, "on_step"):
        agent.on_step(state, action, reward, next_state, debug)

def run_q_learning(env: NimEnv, agent1, agent2, debug=False, catch_invalid_moves=False):
    players = [agent1, agent2]
    #print("New game")
    if (debug):
        env.render()

    while not env.end:
        #print("state = %s, player = %d" % (env.heaps, env.current_player))
        try:
            env.step(players[env.current_player].act(env.heaps))
            reward = int(env.end)
        except AssertionError as err:
            if catch_invalid_moves:
                # End the game due to invalid move!
                env.end = True
                env.winner = env.current_player
                env.heaps = [0, 0, 0]
                reward = -1
                #print("Caught invalid move!")
            else:
                # Otherwise make sure to propagate on invalid moves
                raise err
        if (debug):
            env.render()

        if env.num_step > 1:
            call_on_step(players[env.current_player], env.prec_state, env.prec_action, -reward, env.heaps, debug)

    # Last state
    if env.num_step > 0:
        call_on_step(players[env.current_player ^ 1], env.last_state, env.last_action, reward, env.heaps, debug)

    return (-1) ** env.winner