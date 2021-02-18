[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[algorithm]: imgs/algorithm.png "Algorithm"
[graph]: imgs/graph.png "Graph"

# Unity's Reacher Environment Solution with RL

This project is an implementation of Deep Deterministic Policy Gradient Reinforcement learning algorithm to solve Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) v0.4 environment.
This report documents my approach, algorithm used and the results obtained in this implementation.

![Trained Agent][image1]

### Problem Statement

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

### Environment

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment consists of 20 identical agents, each with its own copy of the environment.

To consider this enviroment solved, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Approach: Deep Deterministic Policy Gradient (DDPG)

The environment was solved using an Actor-Critic implementation of Deep Deterministic Policy Gradient (DDPG) Reinforcement Learning algorithm.

The Algorithm uses an Actor-Critic architecture based on DPG algorithm(Silver et al., 2014)
The actor function uses a neural network and determines the appropiate action for the agent based on the current state. The critic function uses another neural network to calculate the expected return using the current state.The actor function is trained using the advantage function. 

The implemented algorithm as taken from the paper:

![Algorithm][algorithm]

More about the algorithm can be found here: [Continuous control with deep reinforcement learning (arxiv.org)](https://arxiv.org/pdf/1509.02971.pdf)

#### Code Implementation and Model

The project's code is distributed into 3 files:
- `Continous_Control.py`
  - Import the python packages.
  - Run the environment with Random action Policy.
  - Train 20 agents using DDPG Algorithm
  - Plot the scores.
- `ddpg_agent.py`
  - Create instance of Actor and Critic Neural Network.
  - Initialize the Replay-Memory Buffer and  Noise using Ornstein-Uhlenbeck's process.
  - Train the Neural Network 10 times using random mini-batches from the Memory Buffer every 20 timesteps.
- `model.py`
  - Implements the Actor Neural Network as follows
    - Input Layer of size equal to dimension of states
    - Linear Layer 1 of size 512 units with ReLU activation
    - Batch Normalisation Layer for Layer 1
    - Linear Layer 2 of size 256 units with ReLU activation
    - Output Linear Layer 3 of size equal to number of actions
  - Implements the Critic Neural Network as follows
    - Input Layer of size equal to dimension of states
    - Linear Layer 1 of size 256+4(number of discrete actions) units with Leaky ReLU activation
    - Batch Normalisation Layer for Layer 1
    - Linear Layer 2 of size 256 units with Leaky ReLU activation
    - Linear Layer 3 of size 128 units with Leaky ReLU activation
    - Output Linear Layer 4 of size 1

#### Hyperparameters
````
BUFFER_SIZE = int(1e6)    # replay buffer size
BATCH_SIZE = 128          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR_ACTOR = 1e-4           # learning rate of the actor 
LR_CRITIC = 1e-3          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay
TRAN_MINI_BATCHES = 10    # Number of Batches to train every 20 time steps
EPSILON = 1.0             # Noise factor
EPSILON_DECAY = 0.999999  # Noise factor decay
````
#### Training optimizations:
- Used Batch Size of 128 with Replay buffer of size 1e6 experiences
- Random Noise to action with Noise decay updated every learn step
- Used Gradient Clipping to the Critic Network using `torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)`

### Results

The environment was solved in 320 episodes after getting an average score of 30.04 for continous 100 episodes.
The plot of rewards per episode is as follows.

![Graph][graph]

### Ideas for future work
The following optimisations can be implemented to improve the performance and accuracy when training simultanously on multiple agents.
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)
- [A3C](https://arxiv.org/pdf/1602.01783.pdf)
- [D4PG](https://openreview.net/pdf?id=SyZipzbCb)

### References
- The Code is based on code implementation of DDPG with OpenAI Gym's BipedalWalker-v2 environment in Udacity's [Deep Reinforcement Learning Nanodegree](https://classroom.udacity.com/nanodegrees/nd893).
- [Continuous control with deep reinforcement learning Paper](https://arxiv.org/pdf/1509.02971.pdf)