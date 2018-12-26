import pickle
import tensorflow as tf
import numpy as np

datafile_path = "expert_data/Hopper-v2.pkl"
env_name = "Hopper-v2"
input_dim = 111
output_dim = 8
batch_size = 100
total_iter = 100000

sess = tf.Session()

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

input_ph, output_ph, output_pred = create_model()

# define loss
loss = tf.losses.mean_squared_error(output_ph, output_pred)
opt = tf.train.AdamOptimizer().minimize(loss)

# load training data
f = open(datafile_path, 'rb')
expert_data = pickle.load(f)
f.close()
expert_data['actions'] = expert_data['actions'].reshape((expert_data['actions'].shape[0], -1))

# initialize variables and prepare for training
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# run training
for step in range(total_iter):
    indices = np.random.randint(low=0, high=len(expert_data['actions']), size=batch_size)
    input_batch = expert_data['observations'][indices]
    output_batch = expert_data['actions'][indices]
    _, currloss = sess.run(
        [opt, loss],
        feed_dict={
            input_ph: input_batch,
            output_ph: output_batch
        }
    )
    if step > 0 and step % 1000 == 0:
        print("{0:-8d} iter, loss: {1:.5f}".format(step, currloss))
        if step % 5000 == 0:
            saver.save(sess, "saved_models/" + env_name + "-" + str(step) + ".ckpt")
