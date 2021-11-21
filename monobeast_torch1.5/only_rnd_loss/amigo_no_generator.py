# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import random
import sys
sys.path.insert(0,'../..')

import argparse
import logging
import os
import threading
import time
import timeit
import traceback
import pprint
import typing
import pickle as pkl
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F


torch.multiprocessing.set_sharing_strategy('file_system')

import gym
import gym_minigrid.wrappers as wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from torch.distributions.normal import Normal

from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from env_utils import Observation_WrapperSetup, FrameStack


# Some Global Variables

# yapf: disable
parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                    help='Gym environment.')
parser.add_argument('--mode', default='train',
                    choices=['train', 'test', 'test_render'],
                    help='Training or test mode.')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')

# Training settings.
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint.')
parser.add_argument('--savedir', default='./experimentsMinigrid',
                    help='Root dir where experiment data will be saved.')
parser.add_argument('--total_frames', default=600000000, type=int, metavar='T',
                    help='Total environment frames to train for.')
parser.add_argument('--num_actors', default=4, type=int, metavar='N',
                    help='Number of actors (default: 4).')
parser.add_argument('--num_buffers', default=None, type=int,
                    metavar='N', help='Number of shared-memory buffers.')
parser.add_argument('--num_threads', default=4, type=int,
                    metavar='N', help='Number learner threads.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')

# Loss settings.
parser.add_argument('--entropy_cost', default=0.0005, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--generator_entropy_cost', default=0.05, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')
parser.add_argument('--reward_clipping', default='abs_one',
                    choices=['abs_one', 'soft_asymmetric', 'none'],
                    help='Reward clipping.')

# Optimizer settings.
parser.add_argument('--learning_rate', default=0.001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=0.01, type=float,
                    help='RMSProp epsilon.')


# Other Hyperparameters
parser.add_argument('--batch_size', default=8, type=int, metavar='B',
                    help='Learner batch size (default: 4).')
parser.add_argument('--unroll_length', default=100, type=int, metavar='T',
                    help='The unroll length (time dimension; default: 64).')
parser.add_argument('--goal_dim', default=10, type=int,
                    help='Size of Goal Embedding')
parser.add_argument('--state_embedding_dim', default=256, type=int,
                    help='Dimension of the state embedding representation used in the student')

# Map Layout 
parser.add_argument('--fix_seed', action='store_true',
                    help='Fix the environment seed so that it is \
                    no longer procedurally generated but rather a layout every time.')
parser.add_argument('--env_seed', default=1, type=int,
                    help='The seed to set for the env if we are using a single fixed seed.')
parser.add_argument('--inner', action='store_true',
                    help='Exlucde outer wall')
parser.add_argument('--num_input_frames', default=1, type=int,
                    help='Number of input frames to the model and state embedding including the current frame \
                    When num_input_frames > 1, it will also take the previous num_input_frames - 1 frames as input.')

# Ablations and other settings
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument('--num_lstm_layers', default=1, type=int,
                    help='Lstm layers.')
parser.add_argument('--disable_use_embedding', action='store_true',
                    help='Disable embeddings.')
parser.add_argument('--no_extrinsic_rewards', action='store_true',
                    help='Only intrinsic rewards.')
parser.add_argument('--intrinsic_reward_coef', default=1.0, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument('--random_agent', action='store_true',
                    help='Use a random agent to test the env.')
parser.add_argument('--novelty', action='store_true',
                    help='Discount rewards based on times goal has been proposed.')
parser.add_argument('--novelty_bonus', default=0.1, type=float,
                    help='Bonus you get for proposing objects if novelty')
parser.add_argument('--novelty_coef', default=0.3, type=float,
                    help='Modulates novelty bonus if novelty')
parser.add_argument('--restart_episode', action='store_true',
                    help='Restart Episode when reaching intrinsic goal.')
parser.add_argument('--modify', action='store_true',
                    help='Modify Goal instead of having to reach the goal')
parser.add_argument('--no_boundary_awareness', action='store_true',
                    help='Remove Episode Boundary Awareness')
parser.add_argument('--generator_target', default=5.0, type=float,
                    help='Mean target for Gassian and Linear Rewards')
parser.add_argument('--target_variance', default=15.0, type=float,
                    help='Variance for the Gaussian Reward')

# Set your initials
parser.add_argument('--initials', type=str, default='anonymous',
                    help='Person who runs the experiment')


# Save environment
parser.add_argument('--save_env', action='store_true',
                    help='Save environment and goal for inspection')
parser.add_argument('--save_every', type=int, default=50000,
                    help='How often you want to save the environment')


# Flag for test
parser.add_argument('--weight_path', default='model.tar',type = str,
                    help='The path of trained student weights for the environment')
parser.add_argument('--record_video', action='store_true',
                    help='Record video of a single episode during test mode')
parser.add_argument('--video_path', default='minigrid_video.mp4',type = str,
                    help='Path for saving the video')


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


# typing.Dict allows to specify the type of keys and values of the dictionary
# Our buffer will have string keys and a list of torch tensors
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_forward_dynamics_loss(pred_next_emb, next_emb):
    """
    Rnd loss
    """
    forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)
    return torch.sum(torch.mean(forward_dynamics_loss, dim=1))

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


def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    episode_state_count_dict: dict,
    initial_agent_state_buffers, flags):
    """Defines and generates IMPALA actors in multiples threads."""

    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.
        gym_env = create_env(flags)# Create environment instance
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        #gym_env = wrappers.FullyObsWrapper(gym_env)

        # Create a stack of frames if the number of input frames is larger than 1
        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)

        # Observation_WrapperSetup turns the environment into one that can setup observation items into torch
        env = Observation_WrapperSetup(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed)
        # Dictionary with first observation
        env_output = env.initial()
        initial_frame = env_output['frame']

        # Initialize state for student
        agent_state = model.initial_state(batch_size=1)

        # Use the agent to act in the environment. unused_state is the unused_object
        # dumped cause we are not using LSTMs
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            # Remove and return an item from the queue
            index = free_queue.get()
            if index is None:
                break
            # Write old rollout end
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            buffers["initial_frame"][index][0, ...] = initial_frame     
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Now update the count dict
            # Flatten the output of the environment
            episode_state_key = tuple(env_output['frame'].view(-1).tolist())
            # Check if already stored and update the number of times it is
            if episode_state_key in episode_state_count_dict:
                episode_state_count_dict[episode_state_key] += 1  # Update count if already there
            else:
                episode_state_count_dict.update({episode_state_key: 1})
            # Episodic discount value
            buffers['episode_state_count'][index][0, ...] = \
                torch.tensor(episode_state_count_dict.get(episode_state_key))

            # Reset the episode state counts when the episode is over
            if env_output['done'][0][0]:  # Simply access the done tensor's entry in the environment
                episode_state_count_dict = dict()

            # Do new rollout
            for t in range(flags.unroll_length):
                aux_steps = 0
                timings.reset()

                if env_output['done'][0] == 1:  # Generate a New Goal when episode finished
                    # Set the frame as the new initial_frame for the next iteration
                    initial_frame = env_output['frame']

                # If agent is still alive in episode, predict action
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")
                # Perform step in environment
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # Update the buffer with the values of the results of the new step
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                buffers["initial_frame"][index][t + 1, ...] = initial_frame


                # Other round of episodic updates
                # Flatten the output of the environment
                episode_state_key = tuple(env_output['frame'].view(-1).tolist())
                # Check if already stored and update the number of times it is
                if episode_state_key in episode_state_count_dict:
                    episode_state_count_dict[episode_state_key] += 1  # Update count if already there
                else:
                    episode_state_count_dict.update({episode_state_key: 1})
                # Episodic discount value
                # So in buffers you already place the reciprocal
                buffers['episode_state_count'][index][t+1, ...] = \
                    torch.tensor(episode_state_count_dict.get(episode_state_key))

                # Reset the episode state counts when the episode is over
                if env_output['done'][0][0]:  # Simply access the done tensor's entry in the environment
                    episode_state_count_dict = dict()

                timings.time("write")
            # Put it back in the full queue of processes
            full_queue.put(index)
        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


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

def learn(
    actor_model, model, random_target_network,
        predictor_network, batch, initial_agent_state, optimizer,
        predictor_optimizer, scheduler, flags, max_steps=100.0, lock=threading.Lock()):
    """Performs a learning (optimization) step for the policy, and for the generator whenever the generator batch is full."""
    with lock:

        # Rnd intirnsic reward like BeBold
        # Get embeddings for the intrinsic reward
        random_embedding_t = random_target_network(batch, next_state=False) \
            .reshape(flags.unroll_length, flags.batch_size, 128)
        predicted_embedding_t = predictor_network(batch, next_state=False) \
            .reshape(flags.unroll_length, flags.batch_size, 128)
        
        random_embedding_tplus1 = random_target_network(batch, next_state=True) \
            .reshape(flags.unroll_length, flags.batch_size, 128)
        predicted_embedding_tplus1 = predictor_network(batch, next_state=True) \
            .reshape(flags.unroll_length, flags.batch_size, 128)

        # Predict the intrinsic reward
        rnd_novelty_tplus1 = torch.norm(predicted_embedding_tplus1.detach() - random_embedding_tplus1.detach(), dim=2, p=2)
        rnd_novelty_t = torch.norm(predicted_embedding_t.detach() - random_embedding_t.detach(), dim=2, p=2)
        mask_intrinsic_reward = batch['episode_state_count'][1:] == 1
        clamped_rnd_novelty = torch.clamp(rnd_novelty_tplus1 - rnd_novelty_t, min = 0, max = None)
        intrinsic_rewards = 0.1*clamped_rnd_novelty*mask_intrinsic_reward

        # Compute rnd loss
        rnd_loss = 0.1 * compute_forward_dynamics_loss(predicted_embedding_tplus1, random_embedding_tplus1.detach())

        # Loading Batch
        # Keep all frames but the first
        next_frame = batch['frame'][1:].float().to(device=flags.device)
        initial_frames = batch['initial_frame'][1:].float().to(device=flags.device)
        done_aux = batch['done'][1:].float().to(device=flags.device)


        # Now launch the action prediction on the batch with gradient required
        learner_outputs, unused_state = model(batch, initial_agent_state)
        # Pick the last value prediction in the entire run by all processes
        bootstrap_value = learner_outputs["baseline"][-1]
        # Remove first observations (initialization) from experience batch
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        # Remove last step of the output by the agent prediction batch
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        # Extract the extrinsic rewards from the experience replay batch
        rewards = batch["reward"]
        
        # Student Rewards
        if flags.no_generator:
            total_rewards = rewards
        elif flags.no_extrinsic_rewards:
            total_rewards = intrinsic_rewards 
        else:
            total_rewards = rewards + intrinsic_rewards

        # Perform reward clipping with chosen technique
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(total_rewards, -1, 1)
        elif flags.reward_clipping == "soft_asymmetric":
            squeezed = torch.tanh(total_rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(total_rewards < 0, 0.3 * squeezed, squeezed) * 5.0
        elif flags.reward_clipping == "none":
            clipped_rewards = total_rewards
        # Discount where not done (end of episode)
        discounts = (~batch["done"]).float() * flags.discounting
        clipped_rewards += 1.0 * (rewards>0.0).float()  

        # The behaviour policy is the one from the rollout, the target policy
        # is the one that learns from the experience batch
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # Student Loss
        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        # The three losses are summed into a final loss
        total_loss = pg_loss + baseline_loss + entropy_loss + rnd_loss

        # Get the returns for the episodes by fetching timesteps flagged as "done"
        episode_returns = batch["episode_return"][batch["done"]]

        if torch.isnan(torch.mean(episode_returns)):
            aux_mean_episode = 0.0
        else:
            aux_mean_episode = torch.mean(episode_returns).item()
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": aux_mean_episode, 
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "gen_rewards": None,  
            "gg_loss": None,
            "mean_intrinsic_rewards": None,
            "mean_episode_steps": None,
            "ex_reward": None,
        }

        scheduler.step()
        optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        # Set a maximum for the values of the parameters of the student
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        nn.utils.clip_grad_norm_(predictor_network.parameters(), 40.0)
        optimizer.step()
        predictor_optimizer.step()
        # Share parameters of the learner with the actor model (the one performing rollouts)
        actor_model.load_state_dict(model.state_dict())
        return stats


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
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        initial_frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        carried_col =dict(size=(T + 1,), dtype=torch.int64),
        carried_obj =dict(size=(T + 1,), dtype=torch.int64),

        episode_state_count=dict(size=(T + 1,), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

"""
TRAIN FUNCTION
"""

def train(flags):  
    """Full training loop."""
    # Set a label to the training ID based on the date
    if flags.xpid is None:
        flags.xpid = "{}_{}_{}".format(flags.env, flags.initials, time.strftime("%m%d-%H%M"))

    # Set up the logging system
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    # Set the number of shared buffers based on the number of actors
    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")

    # Set maximum length of an unroll event and the batch size
    T = flags.unroll_length
    B = flags.batch_size

    # Check if the system has GPU
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # Create the environment of Minigrid
    env = create_env(flags)

    # If we train with multiple consequent frames, replace env with a stack object
    #env = wrappers.FullyObsWrapper(env)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)

    # Initiate the random and predictor networks
    random_target_network = support_generator(env.observation_space.shape).to(device=flags.device)
    predictor_network = support_generator(env.observation_space.shape).to(device=flags.device)

    # Now create the two models: generator_model is the teacher and model is the student
    model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers)

    # Define the size of the logits as the one of the board
    if flags.inner:
        logits_size = (env.width-2)*(env.height-2)
    else:  
        logits_size = env.width * env.height

    # Call create buffers function with well-defined parameters
    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags, env.width, env.height, logits_size)

    # All the processes of the multi-process run will share data from the buffer
    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # Deal with the multi-processing framework
    actor_processes = []
    # Create a multiprocessing spawn object and the relative queues
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    episode_state_count_dict = dict()  # episodic count
    # Generate different actors as data sharing processes and start them
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, buffers,
                 episode_state_count_dict,
                 initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    # Reassigned the Net object to learner_model
    learner_model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers).to(
        device=flags.device
    )

    # Create an extra predictor optimizer
    predictor_optimizer = torch.optim.RMSprop(
        predictor_network.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    # Define optimizer variables for gradient propagation
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )


    def lr_lambda(epoch):
        """Scheduling for alpha"""
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    # Adjust the scheduling of the lambda parameter
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "gen_rewards",  
        "gg_loss",
        "mean_intrinsic_rewards",
        "mean_episode_steps",
        "ex_reward",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    # Define frames and stats variables that will be overridden by the batch_and_learn function
    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            # Get a batch of previous experience
            batch, agent_state = get_batch(flags, free_queue, full_queue, buffers,
                initial_agent_state_buffers, timings)
            # Launch the learn function
            stats = learn(model, learner_model,
                          random_target_network, predictor_network, batch,
                          agent_state, optimizer,predictor_optimizer, scheduler, flags, env.max_steps)

            timings.time("learn")
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())
    # Put all processes in the free queue
    for m in range(flags.num_buffers):
        free_queue.put(m)

    # Let multiple threads learn
    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        # Save model weights
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()

        if flags.save_env:
            env_goal_dict = {'frame':torch.zeros(flags.total_frames//flags.save_every,1),
                             'env':torch.zeros((flags.total_frames//flags.save_every,env.width, env.height, 3)),
                             'goal':torch.zeros(flags.total_frames//flags.save_every,1)}
            cp = 0
        
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5) 
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            # Save the environment and the goal
            if flags.save_env:
                if frames // flags.save_every > cp:
                    env_goal_dict['env'][cp] = buffers['frame'][-1][-1]
                    env_goal_dict['goal'][cp] = buffers['goal'][-1][-1]
                    env_goal_dict['frame'][cp] = frames
                    cp += 1

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "After %i frames: loss %f @ %.1f fps. %sStats:\n%s",
                frames,
                total_loss,
                fps,
                mean_return,
                pprint.pformat(stats),
            )

        # Dump a pickle with the environments if save_env is true
        if flags.save_env:
            with open(os.path.join(flags.savedir, flags.xpid, 'frames_goals.pkl'), 'wb') as file:
                pkl.dump(env_goal_dict, file)

    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d frames.", frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def test(flags):
    """
    Function that tests the trained agent in a randomly generated environment
    """
    # Choose the device where to save the weights
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    # Create the environment and initialize it
    gym_env = create_env(flags)
    env = Observation_WrapperSetup(gym_env)
    env_output = env.initial()
    initial_frame = env_output['frame']
    done = False

    # If the video is to be recorded, set up what is needed
    if flags.record_video:
        video_recorder = None
        video_recorder = VideoRecorder(gym_env, flags.video_path, enabled = True)

    # Create the agent and load the model
    student = MinigridNet(gym_env.observation_space.shape, gym_env.action_space.n,
                             state_embedding_dim=flags.state_embedding_dim,
                             num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm,
                             num_lstm_layers=flags.num_lstm_layers)
    state_dict = torch.load(flags.weight_path, map_location=torch.device(flags.device))
    student.load_state_dict(state_dict['model_state_dict'])

    # Let teacher and student do a first step in the environment
    agent_state = student.initial_state(batch_size=1)
    agent_output, unused_state = student(env_output, agent_state)

    # First step prediction
    while not done:
        # Perform step
        with torch.no_grad():
            agent_output, agent_state = student(env_output, agent_state)

        gym_env.render()
        if flags.record_video:
            video_recorder.capture_frame()

        env_output = env.step(agent_output["action"])
        done = env_output['done']
    video_recorder.close()
    video_recorder.enabled = False
    return state_dict


def init(module, weight_init, bias_init, gain=1):
    """Global function initializing the weights and the bias of a module"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

"""
The classes implementing the teacher and student nets
"""


class MinigridNet(nn.Module):
    """Constructs the Student Policy which takes an observation and a goal and produces an action."""
    def __init__(self, observation_shape, num_actions, state_embedding_dim=256, num_input_frames=1, use_lstm=False, num_lstm_layers=1):
        super(MinigridNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions  # The total number of actions to do
        self.state_embedding_dim = state_embedding_dim
        self.use_lstm = use_lstm
        self.num_lstm_layers = num_lstm_layers

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        # Same process as the teacher but add a 1 for goal layer
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) * num_input_frames
        
        if flags.disable_use_embedding:
            print("not_using_embedding")
            self.num_channels = (3+1+1+1+1)*num_input_frames

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

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

    def forward(self, inputs, core_state=()):
        """Main Function, takes an observation and a goal and returns and action."""

        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]

        x = x.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
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

# Support teacher embedding the state
class FullObsMinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(FullObsMinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim)

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0] * self.observation_shape[1] + 1,
                                            self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        ##Because Fully_observed
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
        # indices = torch.tensor([i for i in range(x.shape[3]) if i%3==id])
        # object_ids = torch.index_select(x, 3, indices)
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:, :, :, id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:, :, :, id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:, :, :, id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def agent_loc(self, frames):
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:, :, :, 0]
        agent_location = (agent_location == 10).nonzero()  # select object id
        agent_location = agent_location[:, 2]
        agent_location = agent_location.view(T, B, 1)
        return agent_location

    def forward(self, inputs, next_state=False):
        # -- [unroll_length x batch_size x height x width x channels]
        if next_state:
            x = inputs["frame"][1:]
        else:
            x = inputs["frame"][:-1]
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        if next_state:
            agent_loc = self.agent_loc(inputs["frame"][1:])
            carried_col = inputs["carried_col"][1:]
            carried_obj = inputs["carried_obj"][1:]
        else:
            agent_loc = self.agent_loc(inputs["frame"][:-1])
            carried_col = inputs["carried_col"][:-1]
            carried_obj = inputs["carried_obj"][:-1]

        x = x.long()
        agent_loc = agent_loc.long()
        carried_obj = carried_obj.long()
        carried_col = carried_col.long()
        # -- [B x H x W x K]
        x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim=3)
        agent_loc_emb = self._select(self.embed_agent_loc, agent_loc)
        carried_obj_emb = self._select(self.embed_object, carried_obj)
        carried_col_emb = self._select(self.embed_color, carried_col)

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        # -- [B x K x W x H]

        agent_loc_emb = agent_loc_emb.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1)

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        union = torch.cat([x, agent_loc_emb, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        return core_input


# Global variables for teacher and students
Net = MinigridNet
support_generator = FullObsMinigridStateEmbeddingNet



"""
create_env simply instantiates an object of type Minigrid2Image. The latter
is a sub-class of the gym ObservationWrapper, which is a specific class of 
the gym environment that allows to override the way gym returns the environmnet
variables. In our case, the method observation of the Minigrid2Image class
allows to output images after a step. 
"""


class Minigrid2Image(gym.ObservationWrapper):
    """Get MiniGrid observation to ignore language instruction."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, observation):
        return observation["image"]


def create_env(flags):
    return Minigrid2Image(wrappers.FullyObsWrapper(gym.make(flags.env)))


def main(flags):
    """Call the train or test function"""
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
