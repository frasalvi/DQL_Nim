import numpy as np
import random
from operator import itemgetter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

from agent import Agent
from qlearning import get_possible_actions

# General hyperparameters
UPDATE_TARGET_EVERY = 500

class DQNAgent(Agent):
    '''
    Agent implementing Deep Q-learning for the game Nim.

    Parameters:
    ----------
    epsilon: float, in [0, 1]. 
        epsilon of the epsilon-greedy policy.
    alpha: float. 
        learning rate.
    gamma: float. 
        discount factor.
    buffer_size : int.
        maximum length of the replay buffer.
    batch_size : int
        size of the minibatch.
    '''
    def __init__(self, epsilon, alpha=5e-4, gamma=0.99, buffer_size=10_000, batch_size=64):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.model = DQNAgent._create_q_model()
        self.model_target = DQNAgent._create_q_model()
        # the dequeue will automatically evict old entries to avoid exceeding max length
        self.replay_buffer = deque(maxlen=buffer_size) #ReplayBuffer()
        self.batch_size = batch_size

        # use loss and optimizer suggested in the project statement
        self.loss_function = keras.losses.Huber(delta=1)
        self.optimizer = keras.optimizers.Adam(learning_rate=alpha)#, clipnorm=1.0)

        self.last_state = None
        self.last_move = None

    @staticmethod
    def _create_q_model():
        '''
        Creates a deep network for the actor playing Nim.

        Returns
        ----------
        model : keras.Model
            actor playing Nim. the model takes as input
            a 9-bit encoding of the current state and outputs a list
            of 21 probabilities, one for each action.
        '''
        inputs = layers.Input(shape=(3, 3))
        # flatten since Dense expects the previous layer to have dimension 1, otherwise weird stuff happens
        layer0 = layers.Flatten(input_shape=(3, 3), name="flatten")(inputs)
        layer1 = layers.Dense(128, activation="relu", name="dense_1")(layer0)
        layer2 = layers.Dense(128, activation="relu", name="dense_2")(layer1)
        action = layers.Dense(21, activation="linear", name="output_linear")(layer2)

        return keras.Model(inputs=inputs, outputs=action)

    @staticmethod
    def _encode_heaps(heaps):
        '''
        Encodes the current state in a 9-bits representation.

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.
    
        Returns
        ----------
        heaps_converted : tf.Tensor
            9-bit encoding of the current state.
        '''
        heaps_converted = tf.convert_to_tensor([
            [(h & 4) > 0, (h & 2) > 0, (h & 1) > 0] for h in heaps
        ])
        return heaps_converted
    
    @staticmethod
    def _decode_heaps(heaps_converted):
        '''
        Decodes the current state from its 9-bits representation.

        Parameters
        ----------
        heaps_converted : tf.Tensor
            9-bit encoding of the current state.
    
        Returns
        ----------
        heaps : list of integers
            list of heap sizes.
        '''
        heaps = [
            int(row[0]) * 4 + int(row[1]) * 2 + int(row[2]) for row in heaps_heaps
        ]
        return heaps

    def get_qvalues(self, heaps):
        '''
        Get the Q-values of all the 21 actions, for a given state.

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.

        Returns
        ----------
        TODO
        '''
        qvalues = self.model(
            tf.expand_dims(DQNAgent._encode_heaps(heaps), axis=0),
            training=False
        )
        return qvalues[0]

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
    
        """
        # Pick first in case of tie, shouldn't be too bad
        qvls = self.get_qvalues(heaps)
        max_index = np.argmax(qvls)
        return qvls[max_index], (max_index // 7 + 1, max_index % 7 + 1)
        """
        qvls = self.get_qvalues(heaps)
        qvalues_indexed = [(i, qvl) for i, qvl in enumerate(qvls)]
        random.shuffle(qvalues_indexed)
        max_index = max(qvalues_indexed, key=itemgetter(1))[0]
        return qvls[max_index], (max_index // 7 + 1, max_index % 7 + 1)

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
            move = random.choice(get_possible_actions(heaps))
        else:
            # greedy
            move = self._pick_best_move(heaps)[1]
        return move

    def on_step(self, state, action, reward, next_state, debug):
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
        # Convert the action from env format (1-3, 1-7) to internal format 0-20
        internal_action = (action[0] - 1) * 7 + (action[1] - 1)
        # Update replay buffer
        self.replay_buffer.append((
            DQNAgent._encode_heaps(state),
            internal_action,
            reward,
            DQNAgent._encode_heaps(next_state)
        ))

        # Skip updating if we haven't gathered enough samples yet
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample minibatch
        minibatch_indices = np.random.choice(len(self.replay_buffer), size=self.batch_size)
        minibatch = [self.replay_buffer[i] for i in minibatch_indices]
        minibatch_states = np.array([prev[0] for prev in minibatch])
        minibatch_actions = [prev[1] for prev in minibatch]
        minibatch_rewards = [prev[2] for prev in minibatch]
        minibatch_next_states = np.array([prev[3] for prev in minibatch])

        # Compute target
        target_q_values = self.model_target(minibatch_next_states)
        target_term = minibatch_rewards + self.gamma * tf.reduce_max(
            target_q_values,
            axis=1
        )
        mask = tf.one_hot(minibatch_actions, 21)
        with tf.GradientTape() as tape:
            q_values = self.model(minibatch_states)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            self.loss = self.loss_function(target_term, q_action)

        # Backprop
        grads = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target(self):
        self.model_target.set_weights(self.model.get_weights())
