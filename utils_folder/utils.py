# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import typing
import sys
sys.path.insert(0,'../..')
import threading
import torch
from torch import multiprocessing as mp
from torch.nn import functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import gym
import gym_minigrid.wrappers as wrappers

# typing.Dict allows to specify the type of keys and values of the dictionary
# Our buffer will have string keys and a list of torch tensors
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    """
    Compute the loss on the value network
    """
    # Take the mean over batch, sum over time.
    return 0.5 * torch.sum(torch.mean(advantages ** 2, dim=1))


def compute_entropy_loss(logits):
    """
    Compute the regularization entropy loss
    """
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    """
    Compute the policy gradient loss
    """
    # Main Policy Loss
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))

def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock()):
    """Returns a Batch with the history."""
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")

    # Put together a batch of experiences from all the processes
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    # Just for RNN usage
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    # Put free indices back into the free_queue object
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    # Store batch object into the GPU device (if any)
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in initial_agent_state)
    timings.time("device")
    # Return a batch with combined experiences over the last unrolling from all agents
    return batch, initial_agent_state


def reached_goal_func(frames, goals, initial_frames = None, done_aux = None):
    """Auxiliary function which evaluates whether agent has reached the goal."""
    if flags.modify:
        new_frame = torch.flatten(frames, 2, 3)
        old_frame = torch.flatten(initial_frames, 2, 3)
        ans = new_frame == old_frame
        ans = torch.sum(ans, 3) != 3  # reached if the three elements are not the same
        reached = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goals.long(),2)))
        if flags.no_boundary_awareness:
            reached = reached.float() * (1 - done_aux.float())
        return reached
    else:
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0]
        agent_location = (agent_location == 10).nonzero()  # If agent on this tile
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(goals.shape)
        return (goals == agent_location).float()




def create_buffers(obs_shape, num_actions, flags, width, height, logits_size) -> Buffers:
    """Imports characteristics of the state and action space and returns a
    typing.Dict object"""
    T = flags.unroll_length # Buffer time steps length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        generator_baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        generator_logits=dict(size=(T + 1, logits_size), dtype=torch.float32),
        goal=dict(size=(T + 1,), dtype=torch.int64),
        initial_frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        carried_col =dict(size=(T + 1,), dtype=torch.int64),
        carried_obj =dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

# For Minigrid environment creation
class Minigrid2Image(gym.ObservationWrapper):
    """Get MiniGrid observation to ignore language instruction."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, observation):
        return observation["image"]


def create_env(flags):
    return Minigrid2Image(wrappers.FullyObsWrapper(gym.make(flags.env)))