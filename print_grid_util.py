import numpy as np
import torch
import gym
import gym_minigrid
import gym_minigrid.window
import pickle as pkl
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='Print frames')

parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                    help='Gym environment.')

parser.add_argument('--frames_path', type=str, default='.',
                    help='Path of the frames')

parser.add_argument('--frame_to_render', type=int, default=0,
                    help='What frame and goal to print')

flags = parser.parse_args()

# Create the environment 
env = gym.make(flags.env)

# Open the frame file
file = open(flags.frames_path, 'rb')
a = pkl.load(file)
file.close()

# Get a specific frame of interest
frame = np.array(a['frame'][flags.frame_to_render])
array = np.array(a['env'][flags.frame_to_render])
goal = np.array(a['goal'][flags.frame_to_render])

# Create a grid object
grid = gym_minigrid.minigrid.Grid

# Find the position of the agent 
agent_pos = np.where(array[:,:,0] == 10)
agent_pos_list = [agent_pos[0][0],agent_pos[1][0]]

# Strip the agent from it
array[agent_pos] = [[1,5,2]]

print(env.__dict__.keys())

# Create the grid to represent and update the environment with the agent postion
print(goal)
print(frame)
grid_dec = grid.decode(array)
env.grid = grid_dec[0]
env.agent_pos = agent_pos_list
env.mission = ''
img = env.render()

# Get rendering image
im = Image.fromarray(img)
im.save("frame_img.jpeg")



