# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'../../utils_folder')

from models import models
from utils import compute_baseline_loss, compute_entropy_loss, compute_policy_gradient_loss, get_batch, reached_goal_func, create_buffers, Minigrid2Image, create_env

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
# We start t* at 7 steps.
generator_batch = dict()
generator_batch_aux = dict()
generator_current_target = 7.0
generator_count = 0

# typing.Dict allows to specify the type of keys and values of the dictionary
# Our buffer will have string keys and a list of torch tensors
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

# Global variables for teacher and students
Net = models.MinigridNet
GeneratorNet = models.Generator

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)



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
parser.add_argument('--generator_learning_rate', default=0.002, type=float,
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
parser.add_argument('--generator_batch_size', default=32, type=int, metavar='BB',
                    help='Learner batch size (default: 4).')
parser.add_argument('--unroll_length', default=100, type=int, metavar='T',
                    help='The unroll length (time dimension; default: 64).')
parser.add_argument('--goal_dim', default=10, type=int,
                    help='Size of Goal Embedding')
parser.add_argument('--state_embedding_dim', default=256, type=int,
                    help='Dimension of the state embedding representation used in the student')
parser.add_argument('--generator_reward_negative', default= -0.1, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument('--generator_threshold', default=-0.5, type=float,
                    help='Threshold mean reward for wich scheduler increases difficulty')
parser.add_argument('--generator_counts', default=10, type=int,
                    help='Number of time before generator increases difficulty')
parser.add_argument('--generator_maximum', default=100, type=float,
                    help='Maximum difficulty')                    
parser.add_argument('--generator_reward_coef', default=1.0, type=float,
                    help='Coefficient for the generator reward')

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
parser.add_argument('--no_generator', action='store_true',
                    help='Use vanilla policy-deprecated')
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
parser.add_argument('--generator_loss_form', type=str, default='threshold',
                    help='[threshold,dummy,gaussian, linear]')
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

def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    generator_model,
    buffers: Buffers,
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

        # Get the goal from the teacher
        generator_output = generator_model(env_output)
        goal = generator_output["goal"]

        # Use the agent to act in the environment. unused_state is the unused_object
        # dumped cause we are not using LSTMs
        agent_output, unused_state = model(env_output, agent_state, goal)
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
            for key in generator_output:
                buffers[key][index][0, ...] = generator_output[key]   
            buffers["initial_frame"][index][0, ...] = initial_frame     
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                aux_steps = 0
                timings.reset()

                if flags.modify:  # Will probably be False
                    new_frame = torch.flatten(env_output['frame'], 2, 3)
                    old_frame = torch.flatten(initial_frame, 2, 3)
                    ans = new_frame == old_frame
                    ans = torch.sum(ans, 3) != 3  # Reached if the three elements of the frame are not the same.
                    reached_condition = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goal.long(),2)))
                    
                else:
                    #Check if goal was reached
                    agent_location = torch.flatten(env_output['frame'], 2, 3)
                    agent_location = agent_location[:,:,:,0] 
                    agent_location = (agent_location == 10).nonzero() # select object id
                    agent_location = agent_location[:,2]
                    agent_location = agent_location.view(agent_output["action"].shape)
                    reached_condition = goal == agent_location

                if reached_condition:   # Generate new goal when reached intrinsic goal
                    if flags.restart_episode:
                        env_output = env.initial() 
                    else:
                        env.episode_step = 0  # Reset the number of steps in the episode
                    initial_frame = env_output['frame']
                    # Inference, so no requirement of the gradient
                    with torch.no_grad():
                        # Generate new goal
                        generator_output = generator_model(env_output)
                    goal = generator_output["goal"]

                if env_output['done'][0] == 1:  # Generate a New Goal when episode finished
                    # Set the frame as the new initial_frame for the next iteration
                    initial_frame = env_output['frame']
                    with torch.no_grad():
                        generator_output = generator_model(env_output)
                    goal = generator_output["goal"]

                # If agent is still alive in episode, predict action
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state, goal)

                timings.time("model")
                # Perform step in environment
                env_output = env.step(agent_output["action"])

                timings.time("step")

                # Update the buffer with the values of the results of the new step
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                for key in generator_output:
                    buffers[key][index][t + 1, ...] = generator_output[key]  
                buffers["initial_frame"][index][t + 1, ...] = initial_frame     

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

def learn(
    actor_model, model, actor_generator_model, generator_model, batch, initial_agent_state, optimizer,
        generator_model_optimizer, scheduler, generator_scheduler, flags, max_steps=100.0, lock=threading.Lock()):
    """Performs a learning (optimization) step for the policy, and for the generator whenever the generator batch is full."""
    with lock:
        # Loading Batch
        # Keep all frames but the first
        next_frame = batch['frame'][1:].float().to(device=flags.device)
        initial_frames = batch['initial_frame'][1:].float().to(device=flags.device)
        done_aux = batch['done'][1:].float().to(device=flags.device)
        # Get, for each training frame whether the goal was reached or not
        reached_goal = reached_goal_func(next_frame, batch['goal'][1:].to(device=flags.device), initial_frames = initial_frames, done_aux = done_aux)
        # The reward obtained by the agent intrinsically is awarded only when the agent reaches a goal
        intrinsic_rewards = flags.intrinsic_reward_coef * reached_goal
        reached = reached_goal.type(torch.bool)
        # Implement the penalization described at page 7
        intrinsic_rewards = intrinsic_rewards*(intrinsic_rewards - 0.9 * (batch["episode_step"][1:].float()/max_steps))

        # Now launch the action prediction on the batch with gradient required
        learner_outputs, unused_state = model(batch, initial_agent_state, batch['goal'])
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
        total_loss = pg_loss + baseline_loss + entropy_loss

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
            "generator_baseline_loss": None,
            "generator_entropy_loss": None,
            "mean_intrinsic_rewards": None,
            "mean_episode_steps": None,
            "ex_reward": None,
            "generator_current_target": None,
        }

        if flags.no_generator:
            stats["gen_rewards"] = 0.0,  
            stats["gg_loss"] = 0.0,
            stats["generator_baseline_loss"] = 0.0,
            stats["generator_entropy_loss"] = 0.0,
            stats["mean_intrinsic_rewards"] = 0.0,
            stats["mean_episode_steps"] = 0.0,
            stats["ex_reward"] = 0.0,
            stats["generator_current_target"] = 0.0,

        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        # Set a maximum for the values of the parameters of the student
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        # Share parameters of the learner with the actor model (the one performing rollouts)
        actor_model.load_state_dict(model.state_dict())

        # Generator:
        if not flags.no_generator:
            global generator_batch
            global generator_batch_aux
            global generator_current_target
            global generator_count
            global goal_count_dict

            # Loading Batch
            is_done = batch['done']==1
            # Reached is a variable of bools for all timesteps in all dimensions stating
            # whether the agent reached the goal or not at a certain time T in a batch B
            reached = reached_goal.type(torch.bool)
            if 'frame' in generator_batch.keys():
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][is_done].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][is_done].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][is_done].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][is_done].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.zeros(batch['goal'].shape)[is_done].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][is_done].float().to(device=flags.device)), 0)
                
                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0) 
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][reached].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)
            else:
                generator_batch['frame'] = (batch['initial_frame'][is_done]).float().to(device=flags.device) # Notice we use initial_frame from batch
                generator_batch['goal'] = (batch['goal'][is_done]).to(device=flags.device)
                generator_batch['episode_step'] = (batch['episode_step'][is_done]).float().to(device=flags.device)
                generator_batch['generator_logits'] = (batch['generator_logits'][is_done]).float().to(device=flags.device)
                generator_batch['reached'] = (torch.zeros(batch['goal'].shape)[is_done]).float().to(device=flags.device)
                generator_batch['ex_reward'] = (batch['reward'][is_done]).float().to(device=flags.device)
                generator_batch['carried_obj'] = (batch['carried_obj'][is_done]).float().to(device=flags.device)
                generator_batch['carried_col'] = (batch['carried_col'][is_done]).float().to(device=flags.device)

                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0) 
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][reached].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)

            # Run Gradient step, keep batch residual in batch_aux
            if generator_batch['frame'].shape[0] >= flags.generator_batch_size: # Run Gradient step, keep batch residual in batch_aux
                for key in generator_batch:
                    # Keep only a batch of pre-defined size in the generator_batch and place the rest in the auxiliary one
                    generator_batch_aux[key] = generator_batch[key][flags.generator_batch_size:]
                    generator_batch[key] =  generator_batch[key][:flags.generator_batch_size].unsqueeze(0)

                generator_outputs = generator_model(generator_batch)
                # Get the generator value from the output of the generator model
                generator_bootstrap_value = generator_outputs["generator_baseline"][-1]
                
                # Generator Reward
                def distance2(episode_step, reached, targ=flags.generator_target):
                    """The function implementing the reward system for the agent"""
                    aux = flags.generator_reward_negative * torch.ones(episode_step.shape).to(device=flags.device)
                    aux += (episode_step >= targ).float() * reached
                    return aux             

                if flags.generator_loss_form == 'gaussian':
                    generator_target = flags.generator_target * torch.ones(generator_batch['episode_step'].shape).to(device=flags.device)
                    gen_reward = Normal(generator_target, flags.target_variance*torch.ones(generator_target.shape).to(device=flags.device))
                    generator_rewards = flags.generator_reward_coef * (2 + gen_reward.log_prob(generator_batch['episode_step']) - gen_reward.log_prob(generator_target)) * generator_batch['reached'] -1
                
                elif flags.generator_loss_form == 'linear':
                    generator_rewards = (generator_batch['episode_step']/flags.generator_target * (generator_batch['episode_step'] <= flags.generator_target).float() + \
                    torch.exp ((-generator_batch['episode_step'] + flags.generator_target)/20.0) * (generator_batch['episode_step'] > flags.generator_target).float()) * \
                    2*generator_batch['reached'] - 1

                # This is the standard one described by the article
                elif flags.generator_loss_form == 'dummy':
                    generator_rewards = torch.tensor(distance2(generator_batch['episode_step'], generator_batch['reached'])).to(device=flags.device)

                elif flags.generator_loss_form == 'threshold':
                    generator_rewards = torch.tensor(distance2(generator_batch['episode_step'], generator_batch['reached'], targ=generator_current_target)).to(device=flags.device)

                    # If the mean reward is higher than a pre-defined threshold, we increase the difficulty
                if torch.mean(generator_rewards).item() >= flags.generator_threshold:
                    generator_count += 1
                else:
                    generator_count = 0

                # Increasing difficulty means increasing t*
                if generator_count >= flags.generator_counts and generator_current_target<=flags.generator_maximum:
                    generator_current_target += 1.0
                    generator_count = 0
                    goal_count_dict *= 0.0

                # If we want for the teacher to identify where the environment changes the most (False by default)
                if flags.novelty:
                    frames_aux = torch.flatten(generator_batch['frame'], 2, 3)
                    # Get first dimension representing object
                    frames_aux = frames_aux[:,:,:,0]
                    # Create a zero vector with shape of the goal one from the batch's goal
                    object_ids =torch.zeros(generator_batch['goal'].shape).long()
                    for i in range(object_ids.shape[1]):
                        # What object was at the goal site in a certain frame will be stored at object_ids
                        object_ids[0,i] = frames_aux[0,i,generator_batch['goal'][0,i]]
                        goal_count_dict[object_ids[0,i]] += 1

                    # Add the bonus to the reward in case of novelty
                    bonus = (object_ids>2).float().to(device=flags.device)  * flags.novelty_bonus
                    generator_rewards += bonus 

                if flags.reward_clipping == "abs_one":
                    generator_clipped_rewards = torch.clamp(generator_rewards, -1, 1)

                if not flags.no_extrinsic_rewards:
                    generator_clipped_rewards = 1.0 * (generator_batch['ex_reward'] > 0).float() + generator_clipped_rewards * (generator_batch['ex_reward'] <= 0).float()

                generator_discounts = torch.zeros(generator_batch['episode_step'].shape).float().to(device=flags.device)

                # Contains the number of the goal cell at each step
                goals_aux = generator_batch["goal"]
                if flags.inner:
                    goals_aux = goals_aux.float()
                    goals_aux -= 2 * (torch.floor(goals_aux/generator_model.height))
                    goals_aux -= generator_model.height -1
                    goals_aux = goals_aux.long()

                # Get the same exact vtrace return as the student
                generator_vtrace_returns = vtrace.from_logits(
                    behavior_policy_logits=generator_batch["generator_logits"],
                    target_policy_logits=generator_outputs["generator_logits"],
                    actions=goals_aux,
                    discounts=generator_discounts,
                    rewards=generator_clipped_rewards,
                    values=generator_outputs["generator_baseline"],
                    bootstrap_value=generator_bootstrap_value,
                )   

                # Generator Loss
                gg_loss = compute_policy_gradient_loss(
                    generator_outputs["generator_logits"],
                    goals_aux,
                    generator_vtrace_returns.pg_advantages,
                )

                generator_baseline_loss = flags.baseline_cost * compute_baseline_loss(
                    generator_vtrace_returns.vs - generator_outputs["generator_baseline"]
                )

                generator_entropy_loss = flags.generator_entropy_cost * compute_entropy_loss(
                    generator_outputs["generator_logits"]
                )

                generator_total_loss = gg_loss + generator_entropy_loss +generator_baseline_loss

                intrinsic_rewards_gen = generator_batch['reached']*(1- 0.9 * (generator_batch["episode_step"].float()/max_steps))
                stats["gen_rewards"] = torch.mean(generator_clipped_rewards).item()  
                stats["gg_loss"] = gg_loss.item() 
                stats["generator_baseline_loss"] = generator_baseline_loss.item() 
                stats["generator_entropy_loss"] = generator_entropy_loss.item() 
                stats["mean_intrinsic_rewards"] = torch.mean(intrinsic_rewards_gen).item()
                stats["mean_episode_steps"] = torch.mean(generator_batch["episode_step"]).item()
                stats["ex_reward"] = torch.mean(generator_batch['ex_reward']).item()
                stats["generator_current_target"] = generator_current_target
                
                # Perform the training steps for the generator
                generator_scheduler.step()
                generator_model_optimizer.zero_grad() 
                generator_total_loss.backward()
                
                nn.utils.clip_grad_norm_(generator_model.parameters(), 40.0)
                generator_model_optimizer.step()
                # For the next acting step, load the weights of the optimized generator onto the acting one
                actor_generator_model.load_state_dict(generator_model.state_dict())

                if generator_batch_aux['frame'].shape[0]>0:
                    # Copy the items of generator_batch_aux into generator_batch
                    generator_batch = {key: tensor[:] for key, tensor in generator_batch_aux.items()}
                else:
                    generator_batch = dict()
                
        return stats



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

    # Now create the two models: generator_model is the teacher and model is the student
    generator_model = GeneratorNet(env.observation_space.shape, env.width, env.height, num_input_frames=flags.num_input_frames)
    model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers)

    # Create a global variable as a torch tensor of 11 zeros
    global goal_count_dict
    goal_count_dict = torch.zeros(11).float().to(device=flags.device)

    # Define the size of the logits as the one of the board
    if flags.inner:
        logits_size = (env.width-2)*(env.height-2)
    else:  
        logits_size = env.width * env.height

    # Call create buffers function with well-defined parameters
    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags, env.width, env.height, logits_size)

    # All the processes of the multi-process run will share data from the buffer
    model.share_memory()
    generator_model.share_memory()

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

    # Generate different actors as data sharing processes and start them
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, generator_model, buffers,
                 initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    # Reassigned the Net object to learner_model
    learner_model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers).to(
        device=flags.device
    )
    learner_generator_model = Generator(env.observation_space.shape, env.width, env.height, num_input_frames=flags.num_input_frames).to(device=flags.device)

    # Define optimizer variables for gradient propagation
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    generator_model_optimizer = torch.optim.RMSprop(
        learner_generator_model.parameters(),
        lr=flags.generator_learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        """Scheduling for alpha"""
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    # Adjust the scheduling of the lambda parameter
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_model_optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "gen_rewards",  
        "gg_loss",
        "generator_entropy_loss",
        "generator_baseline_loss",
        "mean_intrinsic_rewards",
        "mean_episode_steps",
        "ex_reward",
        "generator_current_target",
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
            stats = learn(model, learner_model, generator_model, learner_generator_model, batch, agent_state, optimizer, generator_model_optimizer, scheduler, generator_scheduler, flags, env.max_steps)

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
                "generator_model_state_dict": generator_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_model_optimizer_state_dict": generator_model_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "generator_scheduler_state_dict": generator_scheduler.state_dict(),
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
    teacher = Generator(gym_env.observation_space.shape, gym_env.width, gym_env.height,
                        num_input_frames=flags.num_input_frames)
    student = MinigridNet(gym_env.observation_space.shape, gym_env.action_space.n,
                             state_embedding_dim=flags.state_embedding_dim,
                             num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm,
                             num_lstm_layers=flags.num_lstm_layers)
    state_dict = torch.load(flags.weight_path, map_location=torch.device(flags.device))
    student.load_state_dict(state_dict['model_state_dict'])
    teacher.load_state_dict(state_dict['generator_model_state_dict'])

    # Let teacher and student do a first step in the environment
    agent_state = student.initial_state(batch_size=1)
    generator_output = teacher(env_output)
    goal = generator_output["goal"]
    agent_output, unused_state = student(env_output, agent_state, goal)

    # First step prediction
    while not done:
        # Check if the agent has reached the goal
        if flags.modify:
            new_frame = torch.flatten(env_output['frame'], 2, 3)
            old_frame = torch.flatten(initial_frame, 2, 3)
            ans = new_frame == old_frame
            ans = torch.sum(ans, 3) != 3  # Reached if the three elements of the frame are not the same.
            reached_condition = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goal.long(), 2)))
        else:
            agent_location = torch.flatten(env_output['frame'], 2, 3)
            agent_location = agent_location[:, :, :, 0]
            agent_location = (agent_location == 10).nonzero()  # select object id
            agent_location = agent_location[:, 2]
            agent_location = agent_location.view(agent_output["action"].shape)
            reached_condition = goal == agent_location

        # Carry on with the experience if the agent has reached the goal
        if reached_condition:
            if flags.restart_episode:
                env_output = env.initial()
            else:
                env.episode_step = 0
            initial_frame = env_output['frame']
            with torch.no_grad():
                generator_output = teacher(env_output)
            goal = generator_output["goal"]

        # Perform step
        with torch.no_grad():
            agent_output, agent_state = student(env_output, agent_state, goal)

        gym_env.render()
        if flags.record_video:
            video_recorder.capture_frame()

        env_output = env.step(agent_output["action"])
        done = env_output['done']
    video_recorder.close()
    video_recorder.enabled = False
    return


def init(module, weight_init, bias_init, gain=1):
    """Global function initializing the weights and the bias of a module"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


"""
create_env simply instantiates an object of type Minigrid2Image. The latter
is a sub-class of the gym ObservationWrapper, which is a specific class of 
the gym environment that allows to override the way gym returns the environmnet
variables. In our case, the method observation of the Minigrid2Image class
allows to output images after a step. 
"""


def main(flags):
    """Call the train or test function"""
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
