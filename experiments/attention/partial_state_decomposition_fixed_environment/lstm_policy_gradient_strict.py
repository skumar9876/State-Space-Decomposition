import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from environment import World
import pickle

from make_plots import make_plot

######################################################################
# Note: This is an adaptation of Denny Britz's implementation of A2C.#
######################################################################

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "loss_arr", "loss_entropy_arr", "action_entropy_arr", "room_entropy_arr"])

'''
Goal: Attention Mechanism should learn to give subtasks to the agent
and keep it in its field of view as the agent moves down.
Environment is fixed for this.

Attention actions: up, down, do nothing
'''

'''
Network definition functions.
'''
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.01)

    #initial = tf.get_variable(name=name, shape=shape,
    #       initializer=tf.contrib.layers.xavier_initializer())
    #return initial

    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    #learning_rate = 0.0001
    def __init__(self, learning_rate=0.00001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            sentence_size = 4

            # Placeholder for the image input in a given iteration
            self.image_pl = tf.placeholder(tf.float32, [None, 5, 5], name='image')

            self.image = tf.to_float(self.image_pl) / 255.0
            self.image_flat = tf.reshape(self.image, [-1, 5*5])

            # Placeholder for the sentence in a given iteration
            self.sentence = tf.placeholder(tf.float32, [None, sentence_size], name='sentence')

            # LSTM 
            self.lstm_size = 7

            self.lstm_unit = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=False)


            #self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.lstm_size])
            #self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.lstm_size])
            
            #self.lstm_state_input = tf.nn.rnn_cell.LSTMStateTuple(self.initial_lstm_state0,
            #                                                  self.initial_lstm_state1)


            # Placeholder for the current lstm state
            self.lstm_state_input = tf.placeholder(tf.float32, [1, self.lstm_size*2], name='lstm_state')

            ####################
            # Action Variables # 
            ####################

            self.chosen_action = tf.placeholder(tf.int32, [None, 2], name='chosen_action')
            self.chosen_room = tf.placeholder(tf.int32, [None, 2], name='chosen_room')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [None], name='target')

            ##################
            #     Network    #
            ##################

            image_hidden_layer_size = 64*25
            sentence_hidden_layer_size = 32*25
            output_hidden_layer_size = 200

            # Hidden layer for the image input
            # self.W_fc1 = weight_variable([5*5, image_hidden_layer_size])
            # self.b_fc1 = bias_variable([hidden_layer_size])
            # self.h_fc1 = tf.tanh(tf.matmul(self.image_flat, self.W_fc1) + self.b_fc1) #Added tanh activation

            # First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32])
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 5, 5, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)

            self.W_conv2 = weight_variable([3, 3, 32, 64])
            self.b_conv2 = bias_variable([64])

            self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
            self.h_conv2_reshaped = tf.reshape(self.h_conv2, [-1, 25*64])

            # Sentence hidden layer
            self.W_fc_sent = weight_variable([sentence_size, sentence_hidden_layer_size])
            self.b_fc_sent = bias_variable([sentence_hidden_layer_size])
            self.h_fc_sent = (tf.matmul(self.sentence, self.W_fc_sent) + self.b_fc_sent) #ADDED TANH HERE!!!

            # Concatenate the sentence hidden layer output with the h_fc1 output
            self.h_fc1_concatenated = tf.concat_v2([self.h_conv2_reshaped, self.h_fc_sent], 1)
            # self.h_fc1_concatenated = tf.concat_v2([self.h_fc1, self.sentence], 1)

            # Second hidden layer
            # self.W_fc2 = weight_variable([hidden_layer_size + sentence_size, hidden_layer_size])
            self.W_fc2 = weight_variable([image_hidden_layer_size + sentence_hidden_layer_size, output_hidden_layer_size])
            self.b_fc2 = bias_variable([output_hidden_layer_size])

            self.h_fc2 = (tf.matmul(self.h_fc1_concatenated, self.W_fc2) + self.b_fc2) #Added tanh activation

            self.h_fc2_reshaped = tf.reshape(self.h_fc2, [1,-1, output_hidden_layer_size])


            self.step_size = tf.placeholder(tf.int32)
            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])

            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_unit,
                                                            self.h_fc2_reshaped,
                                                            initial_state = self.lstm_state_input,
                                                            sequence_length = self.step_size,
                                                            time_major = False,
                                                            scope = scope)

            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1,7])

            self.action_vals = tf.slice(self.lstm_outputs, [0, 0], [self.step_size, 3])
            self.room_vals = tf.slice(self.lstm_outputs, [0, 3], [self.step_size, 4])


            self.action_probs = tf.squeeze(tf.nn.softmax(self.action_vals))
            self.picked_action_prob = tf.gather_nd(self.action_probs, self.chosen_action)

            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            self.picked_room_prob = tf.gather_nd(self.room_probs, self.chosen_room)

            # Loss and train op
            # self.loss = - (tf.reduce_sum(tf.log(self.picked_action_prob)) + tf.reduce_sum(tf.log(self.picked_room_prob))) * self.target
            # self.loss = - (tf.reduce_sum( (tf.log(self.picked_action_prob) + tf.log(self.picked_room_prob)) * self.target))
            self.loss = - (tf.reduce_sum( tf.multiply(tf.add(tf.log(self.picked_action_prob), tf.log(self.picked_room_prob)), self.target )))

            self.action_entropy = -tf.reduce_mean(tf.reduce_sum(self.action_probs * tf.log(self.action_probs), 1), 0)
            self.room_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1), 0)

            action_tensor = tf.log(self.picked_action_prob)
            room_tensor = tf.log(self.picked_room_prob)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, input_sentence, lstm_state_input, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image_pl: input_image, self.sentence: input_sentence, self.lstm_state_input: lstm_state_input, self.step_size: 1}

        action_probs = sess.run(self.action_probs, feed_dict)
        room_probs = sess.run(self.room_probs, feed_dict)
        lstm_state = sess.run(self.lstm_state, feed_dict)


        return action_probs, room_probs, lstm_state

    def update(self, input_image, input_sentence, target, chosen_action, chosen_room, lstm_state_input, step_size, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image_pl: input_image, self.sentence: input_sentence, self.target: target, self.chosen_action: chosen_action, self.chosen_room: chosen_room, 
        self.lstm_state_input: lstm_state_input, self.step_size: step_size}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        lstm_outputs = self.lstm_outputs.eval(feed_dict)
        action_vals = self.action_vals.eval(feed_dict)
        action_probs = self.action_probs.eval(feed_dict)
        chosen_action = self.chosen_action.eval(feed_dict)
        picked_action_prob = self.picked_action_prob.eval(feed_dict)

        picked_room_prob = self.picked_room_prob.eval(feed_dict)
        loss = self.loss.eval(feed_dict)
        action_entropy = self.action_entropy.eval(feed_dict)
        room_entropy = self.room_entropy.eval(feed_dict)


        '''
        print lstm_outputs
        print ""
        print action_vals
        print ""
        print action_probs
        print len(action_probs)
        print ""
        print chosen_action
        print len(chosen_action)
        print ""
        print picked_action_prob
        print len(picked_action_prob)
        print ""
        '''
        
        return loss, action_entropy, room_entropy


class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    #learning_rate = 0.01
    def __init__(self, learning_rate=0.001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            sentence_size = 4

            # Placeholder for the image input in a given iteration
            self.image_pl = tf.placeholder(tf.float32, [1, 5, 5], name='image')

            self.image = tf.to_float(self.image_pl) / 255.0

            self.image_flat = tf.reshape(self.image, [-1, 5*5])

            # Placeholder for the sentence in a given iteration
            self.sentence = tf.placeholder(tf.float32, [1, sentence_size], name='sentence')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [1], name='target')

            ##################
            #     Network    #
            ##################

            hidden_layer_size = 200

            # Hidden layer for the image input
            self.W_fc1 = weight_variable([5*5, hidden_layer_size])
            self.b_fc1 = bias_variable([hidden_layer_size])
            self.h_fc1 = tf.matmul(self.image_flat, self.W_fc1) + self.b_fc1 

            # Sentence hidden layer
            self.W_fc_sent = weight_variable([sentence_size, hidden_layer_size])
            self.b_fc_sent = bias_variable([hidden_layer_size])
            self.h_fc_sent = tf.matmul(self.sentence, self.W_fc_sent) + self.b_fc_sent 

            # Concatenate the sentence hidden layer output with the h_fc1 output
            self.h_fc1_concatenated = tf.concat_v2([self.h_fc1, self.h_fc_sent], 1)
            # self.h_fc1_concatenated = tf.concat_v2([self.h_fc1, self.sentence], 1)

            # Second hidden layer
            # self.W_fc2 = weight_variable([hidden_layer_size + sentence_size, hidden_layer_size])
            self.W_fc2 = weight_variable([hidden_layer_size * 2, hidden_layer_size])
            self.b_fc2 = bias_variable([hidden_layer_size])

            #self.value_estimate = tf.tanh(tf.matmul(self.h_fc1_concatenated, self.W_fc2) + self.b_fc2) 
            self.h_fc2 = tf.matmul(self.h_fc1_concatenated, self.W_fc2) + self.b_fc2

            # Third hidden layer
            # self.W_fc2 = weight_variable([hidden_layer_size + sentence_size, hidden_layer_size])
            self.W_fc3 = weight_variable([hidden_layer_size, 1])
            self.b_fc3 = bias_variable([1])
            self.value_estimate = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3



            self.loss = tf.reduce_sum(tf.squared_difference(self.value_estimate, self.target))


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, input_sentence, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image, self.sentence: input_sentence}
        value_estimate = sess.run(self.value_estimate, feed_dict)

        return value_estimate

    def update(self, input_image, input_sentence, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.sentence: input_sentence, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss




def reinforce(env, estimator_policy, num_episodes, estimator_value=None):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes), 
        loss_arr=np.zeros(num_episodes), 
        loss_entropy_arr=np.zeros(num_episodes), 
        action_entropy_arr=np.zeros(num_episodes), 
        room_entropy_arr=np.zeros(num_episodes)) 
    
    Transition = collections.namedtuple("Transition", ["image", "sentence", "move_action", "room_selection", "next_image", "next_sentence", "done"])
    
    total_reward = 0
    total_time = 0

    counter = 0
    moving_avg_baseline = 0
    moving_avg_sum = 0
    moving_avg_array = []

    room_probs_arr = []
    attention_action_arr = []
    target_sentence_arr = []


    for i_episode in range(num_episodes):
        
        
        episode = []

        chosen_rooms = []
        chosen_actions = []

        images = []
        sentences = []

        reward_arr = []
        baselines = []

        # Initial state of the LSTM memory.
        initial_state = lstm_state = np.zeros([1, 14])

        episode_room_probs_arr =[]
        episode_attention_action_arr = []


        # Reset the environment
        image, target_sentence = env.reset()

        target_sentence_arr.append(target_sentence)

        print ""
        print "Episode: " + str(i_episode)
        print "target sentence: " + str(target_sentence)
        
        # One step in the environment
        for t in itertools.count():


            image = np.array([image])

            
            # Take a step
            action_probs, room_probs, lstm_state = estimator_policy.predict(image, target_sentence, lstm_state)

            episode_room_probs_arr.append(room_probs)
            episode_attention_action_arr.append(action_probs)

            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)

            chosen_actions.append(chosen_action)
            chosen_rooms.append(chosen_room)

            images.append(image)
            sentences.append(target_sentence)

            next_image, next_target_sentence, reward, done = env.step_pg(chosen_room, chosen_action, strict=True)

            counter += 1
            moving_avg_array.append(reward)
            if counter == 100:
                moving_avg_baseline = sum(moving_avg_array) / 100.
            if counter > 100:
                moving_avg_baseline = (moving_avg_baseline * 100. + reward - moving_avg_array[0]) / 100.
                moving_avg_array.pop(0)

            
            total_reward += reward
            total_time += 1

            if estimator_value != None:
                baselines.append(estimator_value.predict(image, target_sentence))
                reward_arr.append(reward)
            else:
                #baselines.append(float(total_reward) / total_time)
                avg_reward = moving_avg_baseline
                reward_arr.append(reward - avg_reward)

            if i_episode >= num_episodes - 5 or i_episode == 0:
                print action_probs
                print room_probs
            
            # Keep track of the transition
            episode.append(Transition(
              image=image, sentence=target_sentence, move_action=chosen_action, room_selection=chosen_room, next_image=next_image, next_sentence=next_target_sentence, done=done))
            
            # Update statistics
            # stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Print out which step we're on, useful for debugging.
            # print("\rStep {} @ Episode {}/{} ({})".format(
            #        t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
            # sys.stdout.flush()

            if done:
                break
                
            image = next_image

        room_probs_arr.append(episode_room_probs_arr)
        attention_action_arr.append(episode_attention_action_arr)       

        if i_episode == num_episodes - 1 or i_episode == num_episodes - 2:
            print baselines

        #print chosen_actions
        #print chosen_rooms

        # Perform update for this episode
        images = np.array(images)
        images = images.reshape((len(images), 5, 5))

        sentences = np.array(sentences)
        sentences = sentences.reshape(len(sentences), 4)

        chosen_rooms_new = []
        for i in xrange(len(chosen_rooms)):
            chosen_rooms_new.append([i, chosen_rooms[i]])
        chosen_rooms = chosen_rooms_new

        chosen_rooms = np.array(chosen_rooms)
        chosen_rooms = chosen_rooms.reshape(len(chosen_rooms), 2)

        chosen_actions_new = []
        for i in xrange(len(chosen_actions)):
            chosen_actions_new.append([i, chosen_actions[i]])
        chosen_actions = chosen_actions_new

        chosen_actions = np.array(chosen_actions)
        chosen_actions = chosen_actions.reshape(len(chosen_actions), 2)

        lstm_initial_state = np.zeros([1, 14])

        stats.episode_rewards[i_episode] = env.episode_reward()

        # Set target array to have total return from each point to the end of the episode
        target_arr = np.zeros(len(reward_arr))
        for index in xrange(len(target_arr)):
            for index2 in range(index, len(reward_arr)):
                target_arr[index] += reward_arr[index2]

        if estimator_value != None:

            for i in xrange(len(images)):
                # Update estimator value
                estimator_value.update([images[i]], [sentences[i]], [target_arr[i]])

            # Set target to be the advantage
            for i in xrange(len(baselines)):
                target_arr[i] = target_arr[i] - baselines[i]
        else:
            # Set target to be the advantage
            #for i in xrange(len(baselines)):
            #    target_arr[i] = target_arr[i] - baselines[i]
            pass

        loss, action_entropy, room_entropy = estimator_policy.update(images, sentences, target_arr, chosen_actions, chosen_rooms, lstm_initial_state, len(episode))

        stats.loss_arr[i_episode] = loss
        stats.action_entropy_arr[i_episode] = action_entropy
        stats.room_entropy_arr[i_episode] = room_entropy

        if i_episode % 1000 == 0:
            #fig1 = plt.figure()
            #plt.scatter(np.arange(i_episode + 1), episode_lengths)
            #fig1.savefig(str(i_episode) + "_episode_lengths.png", dpi=fig1.dpi)
            make_plot('policy_gradient_adam_00001_normalized_image/episode_lengths/' + str(i_episode) + "_episode_lengths", stats.episode_lengths[0:i_episode])

            with open('policy_gradient_adam_00001_normalized_image/episode_lengths/' + str(i_episode) + '_episode_lengths.pickle', 'wb') as handle:
                pickle.dump(stats.episode_lengths[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/room_probs/' + str(i_episode) + '_room_probs.pickle', 'wb') as handle:
                pickle.dump(room_probs_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/room_entropy/' + str(i_episode) + '_room_entropy.pickle', 'wb') as handle:
                pickle.dump(stats.room_entropy_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/action_probs/' + str(i_episode) + '_action_probs.pickle', 'wb') as handle:
                pickle.dump(attention_action_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/action_entropy/' + str(i_episode) + '_action_entropy.pickle', 'wb') as handle:
                pickle.dump(stats.action_entropy_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/loss/' + str(i_episode) + '_loss.pickle', 'wb') as handle:
                pickle.dump(stats.loss_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/target_sentences/' + str(i_episode) + '_target_sentences.pickle', 'wb') as handle:
                pickle.dump(target_sentence_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
estimator_value = ValueEstimator()

fixed=True
if fixed == True:
    fixed_str = 'fixed'
else:
    fixed_str = 'not_fixed'

env = World(fixed=fixed)


# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.167)

num_episodes = 100000#20000
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, num_episodes, estimator_value=None)


with open('episode_lengths_lstm_policy_gradient_strict_lr_00001.pickle', 'wb') as handle:
    pickle.dump(stats.episode_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('episode_rewards_lstm_policy_gradient_strict_lr_00001.pickle', 'wb') as handle:
    pickle.dump(stats.episode_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_lengths)
fig1.savefig('attention/' + fixed_str + '/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('attention/' + fixed_str + '/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('attention/' + fixed_str + '/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.action_entropy_arr)
fig4.savefig('attention/' + fixed_str + '/action_entropy.png', dpi=fig4.dpi)

fig5 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.room_entropy_arr)
fig5.savefig('attention/' + fixed_str + '/room_entropy.png', dpi=fig5.dpi)