# hungry-monkey
a reinforcement learner for collecting bananas

## The Task
Unity's banana environment is a fairly simple environment which allows a player (human or bot) to move around an environment with the goal of collecting yellow bananas and avoiding blue bananas.  This environment is a useful testing ground for reinforcement learning agents.  

The goal of this project is to design a deep-Q-learning agent that can solve this environment from scratch (i.e., learn by pure experience how to behave in a way that optimizes rewards) in as few episodes as possible.  As detailed in the results, I've combined optimization tricks from several different papers to help my learner achieve this.

## Materials
The primary document in this repository is *Report.ipynb*, which contains a writeup of the methods and results of this project, along with the function calls and parameters used to generate them.  All the code that constitutes the agents is contained in the *base_learner* and *advanced_learner* folders.

## Setup
To replicate the results or otherwise run the code in this repository, you'll need to do the following:
1. Clone the repository
2. Install banana.app
3. Make a new conda environment and install requirements.txt
4. Run the code in Report.ipynb


