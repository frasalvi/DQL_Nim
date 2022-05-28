import matplotlib.pyplot as plt
import numpy as np
import random

from nim_env import NimEnv, OptimalPlayer


def measure_performance(agent, other_epsilon, N=500):
    '''
    Measure performance of a given agent against the OptimalPlayer,
    over N games, switching the initial player at each game.

    Parameters
    ----------
    agent : Agent. 
        agent playing againt the OptimalPlayer.
    other_epsilon : float in [0, 1]. 
        epsilon of the epsilon-greedy policy for the OptimalPlayer.
    N : int. 
        number of games to be played.

    Returns
    -------
    won_ratio : float. 
        ratio of games won by agent.
    '''
    players = [agent, OptimalPlayer(other_epsilon)]
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


def Mopt(agent, N=500):
    '''
    Measure performance of a given agent against the OptimalPlayer with epsilon=0,
    over N games, switching the initial player at each game.

    Parameters
    ----------
    agent : Agent. 
        agent playing againt the OptimalPlayer.
    N : int. 
        number of games to be played.

    Returns
    -------
    mopt : float. 
        ratio of games won by agent.
    '''
    return measure_performance(agent, 0)


def Mrand(agent, N=500):
    '''
    Measure performance of a given agent against the OptimalPlayer with epsilon=1,
    over N games, switching the initial player at each game.

    Parameters
    ----------
    agent : Agent. 
        agent playing againt the OptimalPlayer.
    N : int. 
        number of games to be played.

    Returns
    -------
    mrand : float. 
        ratio of games won by agent.
    '''
    return measure_performance(agent, 1)


def get_possible_actions(heaps):
    '''
    Compute the list of allowed actions for a given state.

    Parameters
    ----------
    heaps : list of integers
        list of heap sizes.

    Returns
    -------
    actions : list
        actions[0] is a list of heaps to take from (starts at 1)
        actions[1] is the number of elements taken from heaps actions[0]
    '''
    actions = []
    for i, heap_size in enumerate(heaps):
        for number in range(1, heap_size+1):
            actions.append((i+1, number))
    return actions


def call_on_step(agent, state, action, reward, next_state, debug):
    '''
    Wrapper for Agent.on_step, allowing to call it only for instances
    of Agent, ignoring it for the OptimalPlayer.

    Parameters
    ----------
    agent : Agent or env.OptimalPlayer. 
        agent for which on_step should be called.
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
    if hasattr(agent, "on_step"):
        agent.on_step(state, action, reward, next_state, debug)


def run_q_learning(env: NimEnv, agent1, agent2, debug=False, catch_invalid_moves=False):
    '''
    Core logic of the game, running a match of Nim until its end and 
    updating parameters of the agents.

    Parameters
    ----------
    env: NimEnv. 
        instance of the environment for the game Nim.
    agent1 : Agent or env.OptimalPlayer.
    agent2 : Agent or env.OptimalPlayer.
    debug : bool. 
        if true, print debug information.
    catch_invalid_moves : bool. 
        if true, invalid moves are caught.
    
    Returns
    -------
    reward : int.
        final reward for agent1.
    '''
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
                break   ## !! we don't want to update the Q-value of the previous state.
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