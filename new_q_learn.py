"""
Authors:  Phil Hopkins
CS 440 Introduction to Artificial Intelligence
Project SWE-RL
Description: Classes/Functions to implement Q learning
"""
"""
The following code has been modified from the original file provided
by https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj6/
The original attribution is kept as a comment below.
"""
# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""
This code is also modified from that turned in for Project RL in CS 440,
Fall 2024 at Colorado State University.
"""

# imports
import random
import sys
import csv
import json

class QLearningAgent():
    """
      Q-Learning Agent that takes inputs of the parameters and actions, and
      is updated based on given (state, action, state, reward) tuples
    """
    def __init__(self, epsilon, alpha, discount, legal_actions):
        """Initializes the Q Learning class with parameters and legal actions"""
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        self.legal_actions = legal_actions

        # Initialize a dictionary, that will eventually be {state:{action: Q}}
        self.q_values = {}

    def get_actions(self, state):
        """Returns the list of all legal actions"""
        return self.legal_actions

    def get_q_value(self, state, action):
        """
          Returns the q value for a state-action pair
          If state has never been seen before, returns 0.0
          and adds it to the list of states
        """
        # If it does not exist, add an entry for each state/action pair, initialized to zero
        if state not in self.q_values:
            actions_to_add = self.get_actions(state)
            action_dict = {}
            for each_action in actions_to_add:
                action_dict[each_action] = 0.0
            self.q_values[state] = action_dict
        return self.q_values[state][action]

    def compute_max_q_value(self, state):
        """
          Returns the maximum q value for all possible actions
          from that state.
          Returns 0 if there are no legal actions
        """
        all_q_values = []
        action_list = self.get_actions(state)
        # Collect all Q values possible for all actions
        for each_action in action_list:
            all_q_values.append(self.get_q_value(state, each_action))
        if len(action_list) == 0:
            return 0.0
        # Return the best Q value
        return max(all_q_values)

    def compute_next_action_from_q_values(self, state):
        """
          Determines the action to take from a state, that
          has the best q-value from that state. If no legal
          actions exist, then it returns None. Ties are broken
          via a random choice.
        """
        action_list = self.get_actions(state)
        if action_list is None or len(action_list) == 0:
            return None
        # Get the best Q possible for the state
        max_action_q = self.compute_max_q_value(state)
        list_of_tie_actions = []
        # Find the action corresponding to that Q
        for each_action in action_list:
            if self.get_q_value(state, each_action) == max_action_q:
                list_of_tie_actions.append(each_action)
        # Returns a random action when there are ties
        return random.choice(list_of_tie_actions)

    def get_action(self, state):
        """
          Returns the action to take, given the current state.
          Based on the epsilon value, a random choice will be chosen
          with probability of epsilon. Otherwise, the best action is chosen
        """
        # Pick Action
        legal_actions = self.get_actions(state)
        action = None

        # Choose randomly if true, otherwise go with the best
        if (random.random() < self.epsilon) is True:
            action = random.choice(legal_actions)
        else:
            action = self.compute_next_action_from_q_values(state)
        return action

    def update(self, state, action, next_state, reward: float):
        """
          Update the q-value using a simple Bellman update,
          taking a state and action, as well as the resulting
          state and the reward, to update the q-value
        """
        # Using pseudocode on Pg. 803 in AIMA textbook, Russel and Norvig, 4e
        q_update = reward + self.discount * self.compute_max_q_value(next_state) - self.get_q_value(state, action)
        self.q_values[state][action] = self.get_q_value(state, action) + self.alpha * q_update

    def get_policy(self, state):
        """
        Returns the best action to take for a state.
        """
        return self.compute_next_action_from_q_values(state)

    def get_value(self, state):
        """
        Returns the best q-value for a given state.
        """
        return self.compute_max_q_value(state)


def main():
    """
    Main method to generate the q-values from the training data input.
    """
    # Parse command line arguments
    csv_file = sys.argv[1] # File name of CSV file with training data
    action_list = []
    # Rest of arguments are the action list
    for each_arg in range(2, len(sys.argv)):
        action_list.append(str(sys.argv[each_arg]))

    # Get the data from the CSV
    data_lists = []
    with open(csv_file) as open_csv:
        csv_reader = csv.reader(open_csv)
        for each_row in csv_reader:
            data_lists.append(each_row)

    NUM_ITER = 25
    # Create new QLearningAgent object
    new_agent = QLearningAgent(0.1, 0.9, 0.9, action_list)
    # Iterate over the entire dataset multiple times, doing an update for each row in the CSV
    for each_iteration in range(NUM_ITER):
        for each_update in data_lists:
            new_agent.update(float(each_update[0]), each_update[1], float(each_update[2]), float(each_update[3]))

    # Print the resulting policy - Can be compared to expected values from the dataset as a "metric"
    print("Resulting Policy, in (Tasks Remaining: Action) format:")
    for each_state in new_agent.q_values:
        print(each_state, ":", new_agent.get_policy(each_state))

    # Save the q-values as JSON
    with open("saved_q_values.txt", "w") as open_file:
        open_file.write(json.dumps(new_agent.q_values))

if __name__ == "__main__":
    main()