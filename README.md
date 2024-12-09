# CS 440 Fall 2024 - Project Reinforcement Learning for SWE
## Phil Hopkins
### Requirements:
Only Python 3 is required for this project. Every package used is included in the base Python3.
### Instructions:
There are three Python scripts that are included. 

create_dataset.py will create a dataset of 30000 entries in the file generated_dataset.csv via the command "create_dataset.py"  
new_q_learn.py will generate a policy from a given dataset and action list via the command "new_q_learn.py dataset_CSV Action1 ... ActionN"  
get_next_action.py will give the next action from the policy from a given state and epsilon value, via the command "get_next_action.py state epsilon Action1 ... ActionN"  

Also included are the generated_dataset.csv and saved_q_values.txt files, which should allow the other programs to work without the initial steps.

### Credit:
This code is modified from the Berkeley CS 188 Zip file (https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj6/), which is used as the basis for the Q-learning.
This code was used as the base for the Q-learning, but the actual implementation was added during the course of the Fall 2024 semester. The generated dataset and other functions for
the processing of the data and scripts for getting the next action in a policy were created by me.
