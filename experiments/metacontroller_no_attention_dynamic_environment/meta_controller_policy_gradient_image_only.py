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
    
    #0.00001
    def __init__(self, learning_rate=0.00001, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [None, 10, 10], name='image')
            self.image = tf.to_float(self.image) / 255.0

            self.image_flat = tf.reshape(self.image, [-1, 10*10])


            ####################
            # Action Variable  # 
            ####################

            self.chosen_room = tf.placeholder(tf.int32, [None, 2], name='chosen_room')

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [None], name='target')

            ##################
            #     Network    #
            ##################
            # First hidden layer
            #self.W_fc1 = weight_variable([100, 1000])
            #self.b_fc1 = bias_variable([1000])
            #self.h_fc1 = tf.tanh(tf.matmul(self.image_flat, self.W_fc1) + self.b_fc1)

            #self.W_fc1_2 = weight_variable([1000, 500])
            #self.b_fc1_2 = bias_variable([500])
            #self.h_conv2 = tf.tanh(tf.matmul(self.h_fc1, self.W_fc1_2) + self.b_fc1_2)


            # First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32]) #CHANGED FROM 32
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)

            # Second convolutional layer
            # self.W_conv2 = weight_variable([3, 3, 32, 64])
            # self.b_conv2 = bias_variable([64])
            # self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)

            # Third convolutional layer
            # self.W_conv3 = weight_variable([3, 3, 64, 64])
            # self.b_conv3 = bias_variable([64])
            # self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3) + self.b_conv3)

            flattened_conv_output = 100*32
            hidden_layer_size = 1000 # CHANGED FROM 500 to 1000
            self.h_pool2_reshaped = tf.reshape(self.h_conv1, [-1, flattened_conv_output])

            # Second hidden layer
            self.W_fc2 = weight_variable([flattened_conv_output, hidden_layer_size])
            self.b_fc2 = bias_variable([hidden_layer_size])
            self.h_fc2 = (tf.matmul(self.h_pool2_reshaped, self.W_fc2) + self.b_fc2) 

            # Third hidden layer
            self.W_fc3 = weight_variable([hidden_layer_size, 4])
            self.b_fc3 = bias_variable([4])
            self.h_fc3 = (tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3) 



 
            self.room_vals = tf.reshape(self.h_fc3, [-1, 4])

            self.room_probs = tf.squeeze(tf.nn.softmax(self.room_vals))
            self.picked_room_prob = tf.gather_nd(self.room_probs, self.chosen_room)

            # Loss and train op
            self.loss = - (tf.reduce_sum( tf.multiply( tf.log(self.picked_room_prob), self.target ) ))

            # Computed for debugging purposes
            self.log_prob = tf.log(self.picked_room_prob)
            self.intermediate_loss = tf.multiply (tf.log(self.picked_room_prob), self.target)

            self.probs = self.room_probs * tf.log(self.room_probs)
            self.prob_loss = -tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1)

            self.room_entropy = -tf.reduce_mean(tf.reduce_sum(self.room_probs * tf.log(self.room_probs), 1), 0)

            #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())



    
    def predict(self, input_image, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image}


        room_probs = sess.run(self.room_probs, feed_dict)

        return room_probs

    def update(self, input_image, target, chosen_room, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.target: target, self.chosen_room: chosen_room}
        

        _, loss, room_entropy = sess.run([self.train_op, self.loss, self.room_entropy], feed_dict)

        return loss, room_entropy

class ValueEstimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            ###################
            # State Variables #
            ###################

            # Placeholder for the image input in a given iteration
            self.image = tf.placeholder(tf.float32, [1, 10, 10], name='image')
            self.image_flat = tf.reshape(self.image, [-1, 10*10])

            ###################
            # Target Variable #
            ###################

            self.target = tf.placeholder(tf.float32, [1], name='target')

            ##################
            #     Network    #
            ##################

            # First convolutional layer
            self.W_conv1 = weight_variable([3, 3, 1, 32])
            self.b_conv1 = bias_variable([32])

            self.image_reshaped = tf.reshape(self.image, [-1, 10, 10, 1])
            self.h_conv1 = tf.nn.relu(conv2d(self.image_reshaped, self.W_conv1) + self.b_conv1)
            #self.h_pool1 = max_pool_2x2(self.h_conv1)


            # Second convolutional layer
            self.W_conv2 = weight_variable([3, 3, 32, 64])
            self.b_conv2 = bias_variable([64])
            self.h_conv2 = tf.nn.relu(conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)
            self.h_pool2_reshaped = tf.reshape(self.h_conv2, [-1, 100*64])

            image_hidden_layer_size = 100*64

            # Second hidden layer
            self.W_fc2 = weight_variable([image_hidden_layer_size, 200])
            self.b_fc2 = bias_variable([200])
            self.h_fc2 = tf.matmul(self.h_pool2_reshaped, self.W_fc2) + self.b_fc2

            # Third hidden layer
            self.W_fc3 = weight_variable([200, 1])
            self.b_fc3 = bias_variable([1])

            self.value_estimate = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3

            self.loss = tf.reduce_sum(tf.squared_difference(self.value_estimate, self.target))


            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    
    def predict(self, input_image, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.image: input_image}
        value_estimate = sess.run(self.value_estimate, feed_dict)

        return value_estimate

    def update(self, input_image, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.image: input_image, self.target: target}
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

    total_reward = 0
    total_time = 0

    counter = 0
    moving_avg_baseline = 0
    moving_avg_sum = 0
    moving_avg_array = []

    room_probs_arr = []
    target_sentence_arr = []
    
    for i_episode in range(num_episodes):
               
        
        episode = []

        chosen_rooms = []

        images = []

        reward_arr = []
        baselines = []

        episode_room_probs_arr =[]

        # Reset the environment
        image, target_sentence = env.reset_no_attention()

        target_sentence_arr.append(target_sentence)

        if i_episode % 100 == 0 or i_episode >= num_episodes - 10:
            print ""
            print "Episode: " + str(i_episode)
            print "target sentence: " + str(target_sentence)
        
        # One step in the environment
        for t in itertools.count():


            image = np.array([image])

            
            # Take a step
            room_probs = estimator_policy.predict(image)

            episode_room_probs_arr.append(room_probs)

            chosen_room = np.random.choice(np.arange(len(room_probs)), p=room_probs)

            chosen_rooms.append(chosen_room)
            images.append(image)

            next_image, next_target_sentence, reward, done = env.step_no_attention(chosen_room)


            
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
                baselines.append(estimator_value.predict(image))
                reward_arr.append(reward)
            else:
                #baselines.append(float(total_reward) / total_time) #average reward per time step
                #baselines.append(float(total_reward) / (i_episode + 1))

                #avg_reward = float(total_reward) / total_time #average reward per time step

                avg_reward = moving_avg_baseline
                reward_arr.append(reward - avg_reward)



            if i_episode == num_episodes - 1 or i_episode >= num_episodes - 10 or i_episode <= 9:
                #print "room probs:"
                print room_probs
            
            
            # Update statistics
            # stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            

            if done:
                break
                
            image = next_image


        # Perform update for this episode
        images = np.array(images)
        images = images.reshape((len(images), 10, 10))

        chosen_rooms_new = []
        for i in xrange(len(chosen_rooms)):
            chosen_rooms_new.append([i, chosen_rooms[i]])
        chosen_rooms = chosen_rooms_new

        chosen_rooms = np.array(chosen_rooms)
        chosen_rooms = chosen_rooms.reshape(len(chosen_rooms), 2)

        stats.episode_rewards[i_episode] = env.episode_reward()


        total_reward_arr = np.zeros(len(reward_arr))
        reinforce_target_arr = np.zeros(len(reward_arr))
        actor_critic_target_arr = np.zeros(len(reward_arr))

        for index in xrange(len(total_reward_arr)):
            for index2 in range(index, len(reward_arr)):
                total_reward_arr[index] += reward_arr[index2]


        if i_episode >= num_episodes - 4:
            print "baselines:"
            print baselines
            print ""
            print "total reward:"
            print total_reward_arr
            print ""

        if estimator_value != None:

            for i in xrange(len(images)):
                # Update estimator value
                estimator_value.update([images[i]], [total_reward_arr[i]])


            # Set target to be the advantage
            for i in xrange(len(baselines)):
                reinforce_target_arr[i] = total_reward_arr[i] - baselines[i]


            #for i in xrange(len(baselines) - 1):
            #    actor_critic_target_arr[i] = reward_arr[i] + 0.95 * baselines[i+1] - baselines[i]

        else:
            # Set target to be the advantage
            '''
            for i in xrange(len(baselines)):
                reinforce_target_arr[i] = total_reward_arr[i] - baselines[i]
            '''
            reinforce_target_arr = total_reward_arr

        # Update the policy estimator


        episode_loss, episode_loss_entropy = estimator_policy.update(images, reinforce_target_arr, chosen_rooms)

        stats.loss_arr[i_episode] = episode_loss
        stats.loss_entropy_arr[i_episode] = episode_loss_entropy


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

            with open('policy_gradient_adam_00001_normalized_image/loss/' + str(i_episode) + '_loss.pickle', 'wb') as handle:
                pickle.dump(stats.loss_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('policy_gradient_adam_00001_normalized_image/target_sentences/' + str(i_episode) + '_target_sentences.pickle', 'wb') as handle:
                pickle.dump(target_sentence_arr[0:i_episode], handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    return stats



tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
estimator_value = ValueEstimator()

fixed=False
if fixed == True:
    fixed_str = 'fixed_conv'
else:
    fixed_str = 'not_fixed_conv'

env = World(fixed=fixed)

num_runs = 10

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.167)

num_episodes = 100000#20000
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = reinforce(env, policy_estimator, num_episodes, estimator_value=None)




with open('episode_lengths_20k_lr_small_moving_average_baseline.pickle', 'wb') as handle:
    pickle.dump(stats.episode_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('episode_rewards_20k_lr_small_moving_average_baseline.pickle', 'wb') as handle:
    pickle.dump(stats.episode_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)



import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_lengths)
fig1.savefig('no_attention2/' + fixed_str + '/episode_lengths.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.episode_rewards)
fig2.savefig('no_attention2/' + fixed_str + '/episode_rewards.png', dpi=fig2.dpi)

fig3 = plt.figure()
plt.scatter(np.arange(num_episodes), stats.loss_arr)
fig3.savefig('no_attention2/' + fixed_str + '/loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
#print stats.loss_entropy_arr
plt.scatter(np.arange(num_episodes), stats.loss_entropy_arr)
fig4.savefig('no_attention2/' + fixed_str + '/room_entropy.png', dpi=fig4.dpi)