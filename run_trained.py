import tensorflow as tf
import numpy as np
import gym

from my_util import *

trained_model_path = "/Users/xuan/Documents/DeepRL/Berkeley-CS294-112-hw1/selected_models/Walker2d-v2-300000.ckpt"
num_rollouts = 10
max_timesteps = 1000
envname = "Walker2d-v2"
input_dim = 17
output_dim = 6

input_ph, _, output_pred = create_model(input_dim, output_dim)

saver = tf.train.Saver()
with tf.Session() as sess:
    # restore trained model
    saver.restore(sess, trained_model_path)
    # get environment
    env = gym.make(envname)
    
    returns = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = obs.reshape((1, -1))
            action = sess.run(output_pred, feed_dict={input_ph: obs})
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if steps >= max_timesteps:
                break
        returns.append(totalr)
    
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

