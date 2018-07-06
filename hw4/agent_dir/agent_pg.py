#from agent_dir.agent import Agent
import tensorflow as tf
import scipy
import gym
import numpy as np
#import matplotlib.pyplot as plt

def preprocess(I, down_scale=True, bin_pic=True):
    I = I[35:195]
    if down_scale:
        I = I[::2, ::2]
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y / 255. 
    
    if bin_pic:  # Turn gray scale to binary scale
        y[y >= 0.5] = 1.
        y[y < 0.5] = 0.
    
    return np.expand_dims(y.astype(np.float32), axis=0)

class Agent_PG():
    def __init__(self, env, args):
    #def __init__(self):
        """
        Initialize every things you need here.
        For example: building your model
        """
        #super(Agent_PG, self).__init__(env)
        
        self.state_size = [80, 80, 1]
        self.action_size = 3
        self.gamma = 0.99
        self.learning_rate = 0.0005

        ## build model graph
        self._build_model()
        ## init global variables and sess
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            saver = tf.train.Saver()
            checkpoint_dir = './pg_models/'
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def conv2d(self, x, W, stride=[1, 2, 2, 1]):
        return tf.nn.conv2d(input=x, filter=W, strides=stride, padding='SAME')
    
    def weights(self, name, shape, std_weight=0.1, b_std=0.0):
        name = name.lower()
        b_name = name.replace('w', 'b')

        W_ = tf.get_variable(name, shape, 
                initializer=tf.contrib.keras.initializers.he_uniform())
        B_ = tf.get_variable(b_name, shape[-1], 
                initializer=tf.constant_initializer(b_std))

        return W_, B_

    def _build_model(self):
        print("building model ...")
        self.input = tf.placeholder(tf.float32, [None, 80, 80], name='state_input')
        self.label_inputs = tf.placeholder(tf.float32, [None, self.action_size], name='rewards')
        f1 = tf.reshape(self.input, [-1, 6400])

        f_w2, f_b2 = self.weights('f_w2', [6400, 512])
        f2 = tf.nn.relu(tf.matmul(f1, f_w2) + f_b2)
        
        f_w3, f_b3 = self.weights('f_w3', [512, 256])
        f3 = tf.nn.relu(tf.matmul(f2, f_w3) + f_b3)

        f_w4, f_b4 = self.weights('f_w4', [256, 3])
        self.output = tf.nn.softmax(tf.matmul(f3, f_w4) + f_b4)
        
        ## define loss
        crossent = tf.reduce_mean(-tf.reduce_sum(self.label_inputs * tf.log(tf.clip_by_value(self.output,1e-15, 10.0)), reduction_indices=[1]))
        self.agent_train = tf.train.AdamOptimizer(self.learning_rate).minimize(crossent)


    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        #self.gradients.append(np.array(y).astype('float32'))
        self.states.append(state)
        self.rewards.append(reward)
    
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        
        return discounted_rewards

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.prev_x = None

    def train(self, sess):
        gradients = np.squeeze(np.array(self.gradients))
        rewards = np.array(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        rewards = rewards - np.mean(rewards)
        rewards = rewards.repeat(self.action_size).reshape([rewards.shape[0], self.action_size])
        gradients *= rewards
        X = np.squeeze(np.array([self.states]))
        
        Y = np.squeeze(np.array([gradients]))
        
        ## train
        input_data = {}
        input_data[self.input] = X
        input_data[self.label_inputs] = Y
        sess.run(self.agent_train, feed_dict=input_data)
        
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        cur_x = preprocess(observation, True, False)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros([1, 80, 80])
        self.prev_x = cur_x
        aprob = self.sess.run(self.output, feed_dict={self.input: x})
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        aa = np.random.random()
        
        action = np.argmax(prob)
        if action== 0:
            action2 = 1
        elif action == 1:
            action2 = 2
        elif action == 2:
            action2 = 3
        return action2

