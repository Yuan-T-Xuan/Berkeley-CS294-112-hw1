import pickle
import tensorflow as tf
import numpy as np
import gym

from my_util import *
import load_policy

init_model_path = "/Users/xuan/Documents/DeepRL/Berkeley-CS294-112-hw1/selected_models/Humanoid-v2-100000.ckpt"
envname = "Humanoid-v2"
init_expert_data_path = "expert_data/Humanoid-v2.pkl"
expert_policy_path = "experts/Humanoid-v2.pkl"
dagger_iter = 15
train_iter = 40000
input_dim = 376
output_dim = 17

# create model, define loss, etc.
input_ph, output_ph, output_pred = create_model(input_dim, output_dim)
loss = tf.losses.mean_squared_error(output_ph, output_pred)
opt = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()

# load init expert data
f = open(init_expert_data_path, 'rb')
p = pickle.load(f)
f.close()
observations = p['observations']
actions = p['actions'].reshape((p['actions'].shape[0], -1))

# load expert policy
expert_policy_fn = load_policy.load_policy(expert_policy_path)

env = gym.make(envname)
mean_reward = list()
std_reward = list()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, init_model_path)
    # start DAgger
    for dagger_step in range(dagger_iter):
        print("DAgger iter " + str(dagger_step))
        # train/fine-tune current model based on the dataset
        # for simplicity, fix batch_size = 100
        for train_step in range(train_iter+1):
            indices = np.random.randint(low=0, high=len(observations), size=100)
            input_batch = observations[indices]
            output_batch = actions[indices]
            _, currloss = sess.run(
                [opt, loss],
                feed_dict={
                    input_ph: input_batch,
                    output_ph: output_batch
                }
            )
            if train_step > 0 and train_step % 2000 == 0:
                print("{0:-8d} iter, loss: {1:.5f}".format(train_step, currloss))
        # generate 3 rollouts using current model
        rewards = list()
        aggr_observations = list()
        for i in range(3):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                aggr_observations.append(obs)
                obs = obs.reshape((1, -1))
                action = sess.run(output_pred, feed_dict={input_ph: obs})
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                env.render()
                # for simplicity, fix max_timesteps = 1000
                if steps >= 1000:
                    break
            rewards.append(totalr)
        print('rewards', rewards)
        print('mean reward', np.mean(rewards))
        print('std of reward', np.std(rewards))
        mean_reward.append(float(np.mean(rewards)))
        std_reward.append(float(np.std(rewards)))
        # query expert policy to get correct actions for collected observations
        aggr_actions = list()
        for obs in aggr_observations:
            action = expert_policy_fn(obs[None,:])[0]
            aggr_actions.append(action)
        # merge aggr data with current data
        observations = np.append(observations, np.array(aggr_observations), axis=0)
        actions = np.append(actions, np.array(aggr_actions), axis=0)

print(mean_reward)
print(std_reward)
