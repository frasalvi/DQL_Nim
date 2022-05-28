class Agent:
    '''
    Description:
        An abstract class, implementing a generic agent playing Nim. 
        Specialized agents need to inherit from this class
    '''
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
        raise NotImplementedError
    
    def on_step(self, state, action, reward, new_state, debug=False):
        '''
        Update Q-values of the agent, after a step of the environment.

        Parameters
        ----------
        state : list of integers
                list of heap sizes.
        action : action[0] is the heap to take from (starts at 1)
                 action[1] is the number of obj taken from heap action[0]
        reward : int. current reward.
        new_state : list of integers
                    list of heap sizes.
        debug : bool. if true, print debug information.
        '''
        raise NotImplementedError
