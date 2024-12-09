"""
Authors:  Phil Hopkins
CS 440 Introduction to Artificial Intelligence
Project SWE-RL
Description: Script to produce next action based on Q-values and given state
"""

# imports
import json
from new_q_learn import QLearningAgent
import sys

# Command line inputs
epsilon = float(sys.argv[2])
tasks_left = float(sys.argv[1])

# Get rest of command line inputs for action list
action_list = []
for each_arg in range(3, len(sys.argv)):
    action_list.append(str(sys.argv[each_arg]))

# Create a new Q-learning agent
new_agent = QLearningAgent(epsilon, 0.9, 0.9, action_list)
# Get saved Q values
with open("saved_q_values.txt", "r") as open_file:
    read_q_values = json.load(open_file)
# Convert the keys in loaded dictionary to floats
float_converted_q_values = {}
for each_key, each_value in read_q_values.items():
    float_converted_q_values[float(each_key)] = each_value
# Replace q-value dictionary in agent
new_agent.q_values = float_converted_q_values

# Print result from getting the action at the state provided
# Result could be compared to the expected value from the policy as a "metric"
print(f"Your next action, based on the tasks remaining of {tasks_left}, is {new_agent.get_action(tasks_left)}")