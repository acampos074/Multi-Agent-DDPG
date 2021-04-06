## Report
### Project description
The goal of this project is to train two agents to play tennis and to keep the ball in play for as long as possible. A reward of +0.1 is provided to an agent if the ball crosses the net, and a reward of -0.01 each time the ball touches the ground or falls out of bounds.  The task is episodic, and the environment is considered solved once the agents obtain an average score of +0.5 over 100 consecutive episodes.

The environment consists of two agents, an observation space of 24 dimensions per agent (which contains the the position and velocity of the ball and racket for three consecutive time steps), and a continuous action space. Each agent outputs an action vector of two dimensions (which corresponds to translation towards (or away) from the net and jumping), and these values range between -1 and 1.

### Learning algorithm

The algorithm used to solve this problem is the Multi-Agent Deep Deterministic Policy Gradient ([MADDPG](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)). This algorithm is described in Figure 1.

![alt-text](https://raw.githubusercontent.com/acampos074/Multi-Agent-DDPG/master/Figures/maddpg_schematic.png)

**Figure 1 | Schematic illustration of the [MADDPG algorithm](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf).** MADDPG is an extension of the DDPG algorithm, where the critic of each agent has access to the actions and observations of all actors. DDPG is a type of actor-critic method. The actor observes a state and determines the best action. The critic observes a state-action pair and determines an action-value function.

Both agents were trained for a fixed number of episodes and a fixed episode length. For a given state, the actor determines the optimal action deterministically.  [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) noise was added to the action output from the local actor network to build an exploration policy. The critic uses a state-action pair (were the action is determined by the actor) to estimate the optimal action-value function. This action-value function is then used to train the actor. Fig. 2 illustrates a schematic representation of this actor-critic interaction.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/actor_critic.png)

**Figure 2 | Schematic illustration of the DDPG algorithm.** DDPG is a type of actor-critic method. The actor observes a state and determines the best action. The critic observes a state-action pair and determines an action-value function.

The algorithm uses three key ideas:
* Experience Replay
* Soft Updates
* Fixed Q-Targets  

Experience replay helps to break correlations from sequential observations. During training, a set of random experiences are fetched from the memory buffer and these are used to break these correlations. To improve the stability of training, two sets of networks were used (e.g. local and target) for the actor and the critic. Fig. 3 summarizes the network architecture used for the actor.

![alt-text](https://raw.githubusercontent.com/acampos074/Multi-Agent-DDPG/master/Figures/actor_maddpg.png)

**Figure 3 | Neural network architecture.** The input to the actor neural network consist of a 1x48 tensor of states, followed by two fully connected hidden layers with 256 and 128 nodes. The output layer consist of a 1x2 tensor of actions. Each hidden layer is followed by a rectified linear unit (ReLU) activation function. The output layer is followed by a hyperbolic tangent activation function. A batch size of 256 was used to compute each stochastic gradient decent update.

The critic network contains two inputs (states and actions). The input state is a 1x48 tensor and the action is a 1x4 tensor. The network has two hidden units:

- Hidden (states*2 , 256) - ReLU
- Hidden (256+2*actions , 128) - ReLU
- Output (128 , 1) - Linear


Next, a soft update approach helps to slowly adjust the weights of the target network using the weights of the local network. Fig. 3 illustrates how each network is updated.

![alt-text](https://raw.githubusercontent.com/acampos074/DDPG-Continuous-Control/master/Figures/actor_critic_local_target.png)

**Figure 4 | Actor-Critic Local and Target Networks.** The target networks use a soft update approach to increase the stability of training. The local actor network minimizes the loss function based on the action-value function determined by the local critic network. The local critic network minimizes the loss function of the mean square error between the target and expected action-value functions.

Lastly, fixed Q-Targets help to break correlations with the targets (i.e. it helps solve the problem of training with a moving target). Thus, the target weights of both actor and critic are updated less often than the weights of their corresponding local networks. Table 1 lists all the parameters used in this implementation.

#### **Table 1 | List of hyperparameters and their values**
| **Hyperparameter**      | **Value** | **Description**     |
| :---        |    :---   |  :--- |
| `BUFFER_SIZE`      | 100000       | Size of the replay memory buffer. [Adam algorithm](https://arxiv.org/abs/1412.6980) (a variant of stochastic gradient decent (SGD) algorithm) updates are sampled from this buffer.    |
| `BATCH_SIZE`   | 256        | Number of training cases used to compute each SGD update.      |
| `GAMMA`   | 0.99        | Gamma is the discount factor used in the Q-learning update.     |
| `TAU`   | 0.001  |  Tau is an interpolation parameter used to update the weights of the target network. |
| `LR_ACTOR`   | 0.0004  |  Learning rate of the actor.  |
|`LR_CRITIC`   | 0.0004  |  Learning rate of the critic. |
| `THETA_NOISE`   | 0.25  | Long-term mean of the [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823).  |
|  `SIGMA_NOISE` |  0.20 | Standard deviation of the [Ornstein-Uhlenbeck process](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823).    |
|`UPDATE_EVERY`   | 4  | The number of actions taken by the agent between successive SGD updates.  |


Fig. 5 illustrates the temporal evolution of the agent's score-per-episode.

![alt-text](https://raw.githubusercontent.com/acampos074/Multi-Agent-DDPG/master/Figures/maddpg_scores.png)

**Figure 5 | Training curve tracking the agent's score.** The average scores (orange line) shows that the agents were able to receive an average score of at least +0.5 at around 5100 episodes.
### Ideas of Future Work
Other ideas to further improve the agent's performance include further fine-tuning the hyperparameters by increasing the learning rate of both the actor and the critic, and increasing the batch size and buffer size to stabilize training.

Another idea to further improve the training efficiency is to modify the MADDPG algorithm to train multiple instances at the same time.
