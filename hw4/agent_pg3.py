#from agent_dir.agent import Agent
import tensorflow as tf
import scipy
import gym
import numpy as np
import matplotlib.pyplot as plt

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    
    return resized.astype(np.float32)
    #return np.expand_dims(resized.astype(np.float32),axis=2)

def judge_ball_state(cur_x, rreward, episode):
    cur_x = np.squeeze(cur_x)
    xx = []
    #yy = []
    yyy = set()
    for i in range(160):
        if cur_x[i, 140] >= 0.5 and cur_x[i, 140] <= 0.7:
            yyy.add(i)

    for cc in yyy:
        for sub in range(2):
            if cur_x[cc, 140-sub] >= 0.8:
                if episode <= 100:
                    rreward = rreward + 0.1
                #else:
                #    rreward = rreward + 0.05
                #print('Yes')
    
    return rreward

def preprocess(I, down_scale=True, bin_pic=True):
    I = I[35:195]
    if down_scale:
        I = I[::2, ::2]
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y / 255. 
    
    if bin_pic:  # Turn gray scale to binary scale
        y[y >= 0.5] = 1.
        y[y < 0.5] = 0.

    #I[I == 144] = 0
    #I[I == 109] = 0
    #I[I != 0] = 1
    
    return np.expand_dims(y.astype(np.float32), axis=0)

class Agent_PG():
    #def __init__(self, env, args):
    def __init__(self):
        """
        Initialize every things you need here.
        For example: building your model
        """
    #    super(Agent_PG,self).__init__(env)
        
        self.state_size = [80, 80, 1]
        self.action_size = 3
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []

        #if args.test_pg:
            #you can load your model here
        #    print('loading trained model')

    def conv2d(self, x, W, stride=[1, 2, 2, 1]):
        return tf.nn.conv2d(input=x, filter=W, strides=stride, padding='SAME')
    
    def weights(self, name, shape, std_weight=0.1, b_std=0.0):
        name = name.lower()
        b_name = name.replace('w', 'b')

        W_ = tf.get_variable(name, shape, 
        #        initializer=tf.truncated_normal_initializer(stddev=std_weight))
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
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_inputs, logits=f7)
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
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self, sess):
        gradients = np.squeeze(np.array(self.gradients))
        rewards = np.array(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        rewards = rewards - np.mean(rewards)
        rewards = rewards.repeat(self.action_size).reshape([rewards.shape[0], self.action_size])
        gradients *= rewards
        X = np.squeeze(np.array([self.states]))
        
        #Y = np.squeeze(np.array(self.probs)) + self.learning_rate * np.squeeze(np.array([gradients]))
        Y = np.squeeze(np.array([gradients]))
        
        ## train
        input_data = {}
        input_data[self.input] = X
        input_data[self.label_inputs] = Y
        sess.run(self.agent_train, feed_dict=input_data)
        
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def make_action(self, sess, observation, test=True, episode=1):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        #state = observation.reshape([1, observation.shape[0]])
        aprob = sess.run(self.output, feed_dict={self.input: observation})
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        aa = np.random.random()
        #if aa <= 0.01:
        action = np.random.choice(self.action_size, 1, p=prob[0])[0]
        #else:
        #    action = np.argmax(prob)
        
        """
        elif episode <= 1000 and episode > 500 and aa < 0.2:
            action = np.random.choice(self.action_size, 1)[0]
        elif episode > 1000 and episode <= 2000 and aa < 0.1:
            action = np.random.choice(self.action_size, 1)[0]
        elif episode > 2000 and aa < 0.1:
            action = np.random.choice(self.action_size, 1)[0]
        """
        #print(action)
        
        return action, prob

if __name__ == "__main__":
    # start playing environment
    env = gym.make("Pong-v0").unwrapped
    env.seed(11037)
    prev_x = None
    score = 0
    save_scores = []

    state_size = [1, 80, 80]
    action_size = env.action_space.n
    
    # init agent
    agent = Agent_PG()
    agent._build_model()

    saver = tf.train.Saver()
    checkpoint_dir = './models2/best/'
    nn = 0
    max_last30 = 4
    mean_last30 = 0

    with tf.Session() as sess:
        # init global variables
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for episode in range(500):
            state = env.reset()
            while True:
                #if episode % 10 == 0:
                #    env.render()
                cur_x = preprocess(state, True, False)
                x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
                prev_x = cur_x
    
                action, prob = agent.make_action(sess, x, episode)
                if(action == 0):
                    action2 = 1
                elif(action == 1):
                    action2 = 2
                elif(action == 2):
                    action2 = 3
                state, reward, done, info = env.step(action2)
                score += reward

                #cur_x2 = preprocess(state, False, False) 
                #reward = judge_ball_state(cur_x2, reward, episode)  # implement judge ball
                
                agent.remember(x, action, prob, reward)


                if done:
                    agent.train(sess)
                    print('Episode: %d - Score: %f.' % (episode, score))
                    save_scores.append(score)

                    if (len(save_scores)) % 30 == 0:
                        mean_last30 = np.mean(save_scores[-30:])
                        print('Last 30 games mean scores:', mean_last30)
                        
                        with open('./models2/training_scores_test_seed.txt', 'a') as f:
                            for ii,jj in enumerate(save_scores):
                                f.write(str(episode -30 + ii)+':'+str(jj)+'\n')
                        
                        save_scores = []
                        
                        if mean_last30 > max_last30:
                            nn = nn + 1
                            saver.save(sess, './models2/best/PG_best.ckpt')
                            max_last30 = mean_last30
                        
                    score = 0
                    prev_x = None
                    #env.close()
                    break

            if mean_last30 <= -20.90:
                print('The model break QQ')
                break
        saver.save(sess, './models2/policy_3.ckpt')    

