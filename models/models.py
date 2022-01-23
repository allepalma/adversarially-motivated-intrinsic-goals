# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import random
import sys
sys.path.insert(0,'../..')

import torch
from torch import nn
from torch.nn import functional as F


torch.multiprocessing.set_sharing_strategy('file_system')


"""
Weight initializing function
"""
def init(module, weight_init, bias_init, gain=1):
    """Global function initializing the weights and the bias of a module"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


"""
Model for the teacher
"""

class TeacherNet(nn.Module):
    """Constructs the Teacher Policy which takes an initial observation and produces a goal."""
    def __init__(self, observation_shape, width, height, num_input_frames, hidden_dim=256, inner = False):
        super(TeacherNet, self).__init__()
        self.observation_shape = observation_shape
        self.height = height  # Height of grid
        self.width = width  # Width of grid
        self.env_dim = self.width * self.height
        self.state_embedding_dim = 256  # Added to allow training

        self.use_index_select = True  # Used in the select function
        self.obj_dim = 5  # There are 5 types of objects
        self.col_dim = 3  # Three types of colour
        self.con_dim = 2  # Two conditions (door open/door closed)
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) * num_input_frames
        self.inner = inner

        # Initialize the embedding layers for the objects in the environment. They start from a certain vocabulary
        # size (11,6,4) and end with te required dimensionality
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)

        K = self.num_channels  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = 4  # number of convnet layers
        E = 1 # output of last layer

        """
        What you do here is to create a 4-layer CNN representing the teacher.
        It will input a 10-layer input with each level representing one feature
        among possible colour, objects and state. 
        """

        # Make 4 dimensionality preserving convolutional networks
        in_channels = [K] + [M] * 4
        out_channels = [M] * 3 + [E]

        # Create a CNN stack
        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        def interleave(xs, ys):
            """ Return elements of two iterables in interleaved fashion"""
            return [val for pair in zip(xs, ys) for val in pair]

        # Simply place ELU activation after each convolutional layer
        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # The grid size by 16 plus 5 plus 3
        self.out_dim = self.env_dim * 16 + self.obj_dim + self.col_dim

        # Function that initializes the weights of a module as orthogonal matrix and 0 bias
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        # Change the dimensions of the environment if you exclude the outer wall
        if self.inner:
            self.aux_env_dim = (self.height-2) * (self.width-2)
        else:
            self.aux_env_dim = self.env_dim

            # Initialize baseline teacher as a linear layer projecting grid size to 1.
        self.baseline_teacher = init_(nn.Linear(self.aux_env_dim, 1))

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x)  

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def convert_inner(self, goals):
        """Transform environment if using inner flag."""
        goals = goals.float()       
        goals += 2*(1+torch.floor(goals/(self.height-2)))  
        goals += self.height - 1 
        goals = goals.long()
        return goals

    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, height, width, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() # select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)        
        return agent_location 

    def forward(self, inputs):
        """Main Function, takes an observation and returns a goal."""
        x = inputs["frame"]
        T, B, *_ = x.shape
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]


        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)
              
        x = x.transpose(1, 3)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1)

        x = self.extract_representation(x)
        x = x.view(T * B, -1)  # -1 means that the second dimension is inferred by torch

        generator_logits = x.view(T*B, -1)

        generator_baseline = self.baseline_teacher(generator_logits)

        # Sample the index of the goal from a multinomial distribution based on softmax probabilities
        goal = torch.multinomial(F.softmax(generator_logits, dim=1), num_samples=1)

        generator_logits = generator_logits.view(T, B, -1)
        generator_baseline = generator_baseline.view(T, B)
        goal = goal.view(T, B)  # Transform goal to a 1x1 tensor

        if self.inner:
            goal = self.convert_inner(goal)

        return dict(goal=goal, generator_logits=generator_logits, generator_baseline=generator_baseline)


'''
Model for the student 
'''
class StudentNet(nn.Module):
    """Constructs the Student Policy which takes an observation and a goal and produces an action."""
    def __init__(self, observation_shape, num_actions, goal_dim, no_generator = False,
                 state_embedding_dim=256, num_input_frames=1, use_lstm=False, num_lstm_layers=1):
        super(StudentNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions  # The total number of actions to do
        self.state_embedding_dim = state_embedding_dim
        self.use_lstm = use_lstm
        self.num_lstm_layers = num_lstm_layers
        self.no_generator = no_generator

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.goal_dim = goal_dim
        self.agent_loc_dim = 10
        # Same process as the teacher but add a 1 for goal layer
        if not self.no_generator:
            self.num_channels = (self.obj_dim + self.col_dim + self.con_dim + 1) * num_input_frames
        else:
            self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) * num_input_frames

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)
        if not self.no_generator:
            self.embed_goal = nn.Embedding(self.observation_shape[0] * self.observation_shape[1] + 1, self.goal_dim)

        # Weight initialization function
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        # Set up the convolutional net
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )

        # Linear bottleneck
        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.obj_dim + self.col_dim, self.state_embedding_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.state_embedding_dim, self.state_embedding_dim)),
            nn.ReLU(),
        )

        if use_lstm:
            self.core = nn.LSTM(self.state_embedding_dim, self.state_embedding_dim, self.num_lstm_layers)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        # Policy and Baseline are used for baseline subtraction
        self.policy = init_(nn.Linear(self.state_embedding_dim, self.num_actions))
        self.baseline = init_(nn.Linear(self.state_embedding_dim, 1))

    def initial_state(self, batch_size):
        """Initializes LSTM."""
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    # Same functions as in teacher
    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() # select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location    

    def forward(self, inputs, core_state=(), goal=[]):
        """Main Function, takes an observation and a goal and returns and action."""

        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        if not self.no_generator:
            goal = torch.flatten(goal, 0, 1)

            # Creating goal_channel
            goal_channel = torch.zeros_like(x, requires_grad=False)
            goal_channel = torch.flatten(goal_channel, 1,2)[:,:,0]
            for i in range(goal.shape[0]):  # Goal shape is 1 so only one iteration
                goal_channel[i,goal[i]] = 1.0  # Place a 1 only in the board where the goal is
            goal_channel = goal_channel.view(T*B, h, w, 1)  # Put it back to matrix like
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]

        x = x.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        if not self.no_generator:
            goal = goal.long()
            x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2),
                           goal_channel.float()], dim = 3)
        else:
            x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)],
                          dim=3)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 
        union = torch.cat([x, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

'''
Model for random network distillation
'''
class RandomDistillationNetwork(nn.Module):
    def __init__(self, observation_shape):
        super(RandomDistillationNetwork, self).__init__()
        self.observation_shape = observation_shape

        # Initialize the variables describing the environment
        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim)

        # The embedding layers
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0] * self.observation_shape[1] + 1,
                                            self.agent_loc_dim)

        # Weight initialization function
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # Embedding convolutional network
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        # Final FF mapping layers
        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.agent_loc_dim + self.obj_dim + self.col_dim, 128)),
            nn.ReLU(),
            init_(nn.Linear(128, 128)),
            nn.ReLU(),
        )

    def _select(self, embed, x):
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape + (-1,))
        else:
            return embed(x)

    def create_embeddings(self, x, id):
        """ Efficient function to get the embeddings from an index"""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:, :, :, id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:, :, :, id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:, :, :, id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        """Returns the location of the agent from an observation"""
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:, :, :, 0]
        agent_location = (agent_location == 10).nonzero()  # select object id
        agent_location = agent_location[:, 2]
        agent_location = agent_location.view(T, B, 1)
        return agent_location

    def forward(self, inputs, next_state=False):
        """Main state embedding function"""
        # -- [unroll_length x batch_size x height x width x channels]
        if next_state:
            x = inputs["frame"][1:]
        else:
            x = inputs["frame"][:-1]
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        # Out of n total time steps, next_state controls whether we embed states 1 to n-1 (next_state = False) ot
        # states 1 to n (next_state = True)
        if next_state:
            agent_loc = self.agent_loc(inputs["frame"][1:])
            carried_col = inputs["carried_col"][1:]
            carried_obj = inputs["carried_obj"][1:]
        else:
            agent_loc = self.agent_loc(inputs["frame"][:-1])
            carried_col = inputs["carried_col"][:-1]
            carried_obj = inputs["carried_obj"][:-1]

        # Embedding procedure
        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim=3)
        # Select the embeddings for the location, carried object and carried colour from the embedding matrices
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1)

        # Apply the network modules
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        return core_input
