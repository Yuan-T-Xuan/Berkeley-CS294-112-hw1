import tensorflow as tf
import numpy as np
import gym

trained_model_path = "/Users/xuan/Documents/DeepRL/Berkeley-CS294-112-hw1/selected_models/Hopper-v2-90000.ckpt"
num_rollouts = 10
max_timesteps = 1000
envname = "Hopper-v2"
input_dim = 11
output_dim = 3

def create_model():
    global input_dim
    global output_dim
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])
    # variables
    w0 = tf.get_variable(name='w0', shape=[input_dim, 200])
    b0 = tf.get_variable(name='b0', shape=[200])
    w1 = tf.get_variable(name='w1', shape=[200, 200])
    b1 = tf.get_variable(name='b1', shape=[200])
    w2 = tf.get_variable(name='w2', shape=[200, 100])
    b2 = tf.get_variable(name='b2', shape=[100])
    w3 = tf.get_variable(name='w3', shape=[100, output_dim])
    b3 = tf.get_variable(name='b3', shape=[output_dim])
    # define forward propagation
    weights = [w0, w1, w2, w3]
    bias = [b0, b1, b2, b3]
    activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None]
    layer = input_ph
    for w, b, act in zip(weights, bias, activations):
        layer = tf.matmul(layer, w) + b
        if act:
            layer = act(layer)
    # return x, y, y'
    return input_ph, output_ph, layer

input_ph, _, output_pred = create_model()

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

