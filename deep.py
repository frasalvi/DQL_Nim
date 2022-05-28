import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

from qlearning import *

# General hyperparameters
UPDATE_TARGET_EVERY = 500

class DQNAgent:
    '''
    Description:
        A class to implement an epsilon-greedy learning player in Nim.

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.
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
        inputs = layers.Input(shape=(3, 3))
        # flatten since Dense expects the previous layer to have dimension 1, otherwise weird stuff happens
        layer0 = layers.Flatten(input_shape=(3, 3), name="flatten")(inputs)
        layer1 = layers.Dense(128, activation="relu", name="dense_1")(layer0)
        layer2 = layers.Dense(128, activation="relu", name="dense_2")(layer1)
        action = layers.Dense(21, activation="linear", name="output_linear")(layer2)

        return keras.Model(inputs=inputs, outputs=action)

    @staticmethod
    def _encode_heaps(heaps):
        converted = tf.convert_to_tensor([
            [(h & 4) > 0, (h & 2) > 0, (h & 1) > 0] for h in heaps
        ])
        return converted
    
    @staticmethod
    def _decode_heaps(encoded_state):
        converted = [
            int(row[0]) * 4 + int(row[1]) * 2 + int(row[2]) for row in encoded_state
        ]
        return converted

    def get_qvalues(self, heaps):
        # TODO: should it be true in RL training?
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
            max_act : tuple
                max_act[0] is the heap to take from (starts at 1)
                max_act[1] is the number of obj to take from heap #move[0]
        '''
        """
        # Pick first in case of tie, shouldn't be too bad
        qvls = self.get_qvalues(heaps)
        max_index = np.argmax(qvls)
        return qvls[max_index], (max_index // 7 + 1, max_index % 7 + 1)
        """
        best_value = -1e9
        best_moves = [None]
        qvls = self.get_qvalues(heaps)
        max_index = np.argmax(qvls)
        return qvls[max_index], (max_index // 7 + 1, max_index % 7 + 1)
        for heap in range(3):
            for how_much in range(7):
                q_value = qvls[heap * 7 + how_much]
                if q_value >= best_value:
                    if q_value > best_value:
                        best_moves.clear()
                        best_value = q_value
                    best_moves.append((heap + 1, how_much + 1))
        return best_value, random.choice(best_moves)

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
        # Convert the action from env format (1-3, 1-7) to internal format 0-20
        internal_action = (action[0] - 1) * 7 + (action[1] - 1)
        # Update replay buffer
        self.replay_buffer.append((
            DQNAgent._encode_heaps(state),
            internal_action,
            reward,
            DQNAgent._encode_heaps(next_state)
        ))

        #print("adding %s -> %s via action %s, reward %d" % (state, next_state, action, reward))

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

        # Compute gradient
        # Q^(s_j+1)
        target_q_values = self.model_target(minibatch_next_states)
        # max_a' Q^(s_j+1)_a'
        target_term = minibatch_rewards + self.gamma * tf.reduce_max(
            target_q_values,
            axis=1
        )
        mask = tf.one_hot(minibatch_actions, 21)
        with tf.GradientTape() as tape:
            q_values = self.model(minibatch_states)
            # TODO: not sure what exactly these two lines do?
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            loss = self.loss_function(target_term, q_action)

        # Backprop
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target(self):
        self.model_target.set_weights(self.model.get_weights())
