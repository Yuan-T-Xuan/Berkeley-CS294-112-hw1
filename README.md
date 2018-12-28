# How-to
### Behavioral Cloning
1. Run `bash demo.bash` to generate expert data, you may want to change `num_rollouts`.
2. Run `python3 train.py` to train your model. Change `env_name` and `datafile_path` if needed.
3. Run `python3 run_trained.py` to run the trained model. Change `trained_model_path` and `envname` if needed. If env is changed, change `input_dim` and `output_dim` accordingly.

### DAgger
0. An initial model is required to run DAgger, which can be trained using `train.py`.
1. Run `python3 dagger.py`. Change `init_model_path`, `envname`, `init_expert_data_path`, and `expert_policy_path` if needed. If env is changed, change `input_dim` and `output_dim` accordingly.
2. Sit back and watch the robot 'dance'.

### Experiment Result (i.e. Unformatted Report)
[Google Doc](https://docs.google.com/document/d/1dl9_77OlKtTR0dqnZpc-hp_D9eYFlzuidqD1v7uD4fE/edit?usp=sharing)

# (Original README) CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.
