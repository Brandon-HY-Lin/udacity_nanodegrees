
# Deep Reinforcement Learning Nanodegree (DRLND)

## Projects:

1. [Navigation](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/p1_navigation)
	- Purpose: Get rewards when pick up yellow bananna.
	- Framework: Unity and Pytorch
	- Algorithm: DQN (Deep Q-Networks)
	- [Report](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p1_navigation/Report.md)

2. [Continuous Control](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/p2_continuous-control)
	- Purpose: Control multi-joint arms to pick balls.
	- Framework: Unity and Pytorch
	- Algorithm: DDPG (Deep Deterministic Policy Gradients) with PER (Prioritized Experience Replay)
	- [Report](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md)

3. [Collaboration and Competition](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/p3_collab-compet)
	- Purpose: Control 2 table tennis players to play.
	- Framework: Unity and Pytorch
	- Algorithms
		- Algorithm 1:  2 seperate DDPG agents. Each has its own PER buffer ([code](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/p3_collab-compet/DDPG)).
		- Algorithm 2: [MADDPG (Multi-Agent DDPG)](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/p3_collab-compet/MADDPG)
	- [Report](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/Report.md)


## Labs:
* Part 1: Value-Based Method
	* [LunarLander](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/dqn)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: DQN (Deep Q-Networks)

* Part 2: Policy-Based Method
	* [Cartpole](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/hill-climbing)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: Random search best policy weights

	* [Cartpole](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/reinforce)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: REINFORCE algorithm (gradient ascent of expected rewards)

	* [Pong](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/ppo)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: PPO (Proximal Policy Optimization)

	* [Pendulum](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/tree/master/ddpg-pendulum)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: DDPG (Deep Deterministic Policy Gradients)

* Part 3: Multi-Agent Reinforcement
	* [Physical Deception](https://github.com/Brandon-HY-Lin/physical-deception)
		- Framework: OpenAI Gym and Pytorch
		- Algorithm: MADDPG (Multi-Agent DDPG)


	* [Tic-Tac-Toe](https://github.com/Brandon-HY-Lin/udacity-alphazero-Tic-Tac-Toe)
		- Framework: Pytorch
		- Algorithm: AlphaZero
