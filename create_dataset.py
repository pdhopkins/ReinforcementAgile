"""
Authors:  Phil Hopkins
CS 440 Introduction to Artificial Intelligence
Project SWE-RL
Description: Script to create a random dataset for use in Q-learning
"""
# imports
import random

# constants
NUM_ITERATIONS = 1000
FIRST_DIV = 0.5
SECOND_DIV = 0.8
NUM_STEPS = 30
LIVING_REWARD = -0.2


for iteration in range(NUM_ITERATIONS):
    # Go over each step
    for each_step in range(NUM_STEPS, 0, -1):
        # Random chance for which action, and also whether it succeeds
        current_random = random.random()
        decide_to_lower_step = random.random()
        with open("generated_dataset.csv", "a") as open_file:
            to_write = ""
            reward_to_write = LIVING_REWARD
            # "OnlyWork" probability is first third
            if current_random <= 0.33:
                if decide_to_lower_step <= FIRST_DIV:
                    # Make last reward big and positive
                    if each_step == 1:
                        reward_to_write = 15
                    to_write = f"{each_step},OnlyWork,{each_step - 1},{reward_to_write}\n"
                else:
                    to_write = f"{each_step},OnlyWork,{each_step},{reward_to_write}\n"
            # Second third is "Meeting"
            elif current_random > 0.33 and current_random <=0.66:
                if decide_to_lower_step > FIRST_DIV and decide_to_lower_step <= SECOND_DIV:
                    if each_step == 1:
                        reward_to_write = 15
                    to_write = f"{each_step},Meeting,{each_step - 1},{reward_to_write}\n"
                else:
                    to_write = f"{each_step},Meeting,{each_step},{reward_to_write}\n"
            # Final third is "MeetingCodeReview"
            else:
                if decide_to_lower_step >= SECOND_DIV:
                    if each_step == 1:
                        reward_to_write = 15
                    to_write = f"{each_step},MeetingCodeReview,{each_step - 1},{reward_to_write}\n"
                else:
                    to_write = f"{each_step},MeetingCodeReview,{each_step},{reward_to_write}\n"
            open_file.write(to_write)

