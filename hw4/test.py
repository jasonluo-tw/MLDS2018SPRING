import gym
import time
import matplotlib.pyplot as plt
import numpy as np

def preprocess(I):
    I = I[35:195]
    y = 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]
    y = y / 255.
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return y.astype(np.float32)


env = gym.make("Pong-v0")
for i_episode in range(10):
    state = env.reset()
    for t in range(200):
        #env.render()
        #time.sleep(1)
        action = env.action_space.sample()
        #print(action)
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        cur_x = preprocess(state)
        xx = []
        yy = []
        yyy = set()
        for i in range(160):
            if cur_x[i, 140] >= 0.5 and cur_x[i, 140] <= 0.7:
                #print(cur_x[i, j], j, i)
                xx.append(140)  # 140
                yy.append(i)
                yyy.add(i)

        #plt.imshow(cur_x, cmap='gray')
        #plt.plot(xx, yy, 'r*')
        #plt.show()
        #print(yyy)
        for cc in yyy:
            for sub in range(5):
                if cur_x[cc, 140-sub] >= 0.8:
                    print(140-sub, cc)
                    plt.imshow(cur_x, cmap='gray')
                    plt.plot(xx, yy, 'r*')
                    plt.show()

env.close()

#state = env.reset()
#action_size = env.action_space.n
#_,_,_,_ = env.step(0)
#print(env, state.shape, action_size)
