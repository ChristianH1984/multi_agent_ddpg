# Collaboration and Competition: Multi Agent Deep Deterministic Policy Gradient

The repository provides an implementation of a Multi Agent Deep Deterministic Policy Gradient algorithm to solve the 
unity Tennis problem.
There are two agents playing tennis. Each agent receives a positive reward of +0.1 if the ball hits over the net while he receives
a negative reward of 0.01 if he misses or the ball is out of bounds. Hence, this environment can be considered as collaborative 
since both agents try to keep the ball in the game as long as possible.
Each actor receives a state vector of 24 observations and can perform two continuous actions ranging from -1 to 1.
The environment is solved if an average score of 0.5 over hundred consecutive games is reached, 
where average is defined as the average of the maximum score of the agents in each game.
  
 ![tennis](tennis.gif)

## Installation

Download the environment matching your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
 if you need help with determining if your computer is running a 32-bit version 
or 64-bit version of the Windows operating system.

Clone the github repo
- git clone https://github.com/ChristianH1984/multi_agent_ddpg
- cd multi_agent_ddpg
- conda env create --name multi_agent_ddpg --file=environment.yml
- activate multi_agent_ddpg
- Place the uncompressed files into the folder multi_agent_ddpg 
- open Report.ipynb, enter the path to the unity environment in the corresponding notebook cell and enjoy exciting tennis matches :-)
