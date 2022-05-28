from agent import Agent

class QAgent(Agent):
    '''
    Description:
        Agent implementing Q-learning for the game Nim.

    Parameters:
        epsilon: float, in [0, 1]. 
            epsilon of the epsilon-greedy policy.
        alpha: float. 
            learning rate.
        gamma: float. 
            discount factor.
    '''
    def __init__(self, epsilon, alpha=0.1, gamma=0.99):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.qtable = {}

    def get_qvalues(self, heaps):
        '''
        Get the Q-values of all the allowed actions, for a given state.

        Parameters
            ----------
            heaps : list of integers
                list of heap sizes.

        Returns
            ----------
            qtable : dict
                mapping action -> Q-value
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
            best_value: float. 
                highest Q-value for the given state
            best_move : tuple. Action with the highest Q-value for the given state
                best_move[0] is the heap to take from (starts at 1)
                best_move[1] is the number of elements to take from heap best_move[0]
        '''
        # Randomly shuffle the dictionary, in order to randomly break ties.
        qvalues = list(self.get_qvalues(heaps).items())
        random.shuffle(qvalues)
        qvalues = dict(qvalues)

        best_move = None
        for act, val in qvalues.items():
            if best_move is None or val > qvalues[best_move]:
                best_move = act
        return qvalues[best_move], best_move

    def get_max_qvalue(self, heaps):
        '''
        Get the highest Q-value associated to a possible action for the given state.

        Parameters
            ----------
            heaps : list of integers
                list of heap sizes.

        Returns
            ----------
            best_value: float. 
                highest Q-value for the given state.
        '''
        best_value, best_move = self._pick_best_move(heaps)
        # If the game is already finished, return 0.
        return 0 if best_move is None else best_value

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


    def on_step(self, state, action, reward, new_state, debug=False):
        '''
        Update Q-values of the agent with off-policy update, after a step of the environment.

        Parameters
        ----------
        state : list of integers
            list of heap sizes.
        action : list
            action[0] is the heap to take from (starts at 1)
            action[1] is the number of elements taken from heap action[0]
        reward : int. 
            current reward.
        new_state : list of integers
            list of heap sizes.
        debug : bool. 
            if true, print debug information.
        '''
        action = tuple(action)
        old_q_value = self.get_qvalues(state)[action]
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * self.get_max_qvalue(new_state) - old_q_value)
        self.get_qvalues(state)[action] = new_q_value

        if(debug):
            print('Last state: ', state)
            print('Last action: ', action)
            print('Old qvalue: ', old_q_value, '; new qvalue: ', new_q_value, '; reward: ', reward)
