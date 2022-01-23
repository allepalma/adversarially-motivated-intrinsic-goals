# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import sys
sys.path.insert(0,'..')

from models.models import *
from utils_folder.utils import *
from utils_folder.env_utils import Observation_WrapperSetup, FrameStack

import argparse
import logging
import os
import threading
import time
import timeit
import traceback
import pprint
import typing

import torch
from torch import multiprocessing as mp
from torch import nn


torch.multiprocessing.set_sharing_strategy('file_system')
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

# Some Global Variables
# We start t* at 7 steps.
generator_batch = dict()
generator_batch_aux = dict()
generator_current_target = 7.0
generator_count_increment = 0
generator_count_decrement = 0

# typing.Dict allows to specify the type of keys and values of the dictionary
# Our buffer will have string keys and a list of torch tensors
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

# User-defined input variables
parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                    help='Gym environment.')
parser.add_argument('--mode', default='train',
                    choices=['train', 'test', 'test_render'],
                    help='Training or test mode.')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')

# Variables for model to run
parser.add_argument('--model', default='default',
                    choices=['default', 'window_adaptive','novelty_based'],
                    help='Choose the model to run')
parser.add_argument('--window', default=50, type=float,
                    help='The window for the adaptive method')


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
                    help='Entropy cost/multiplier.')  # Student entropy cost
parser.add_argument('--generator_entropy_cost', default=0.05, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')

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
parser.add_argument('--intrinsic_reward_coef', default=1.0, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument('--restart_episode', action='store_true',
                    help='Restart Episode when reaching intrinsic goal.')
parser.add_argument('--modify', action='store_true',
                    help='Modify Goal instead of having to reach the goal')
parser.add_argument('--no_boundary_awareness', action='store_true',
                    help='Remove Episode Boundary Awareness')

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


'''
Act function to perform actions in the environment. A pre-defined set of IMPALA actors parametrized by the learner 
policy collect experience batches in the environment.
'''
def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    generator_model,
    buffers: Buffers,
    episode_state_count_dict : dict,
    initial_agent_state_buffers, flags):
    """Defines and generates IMPALA actors in multiples threads."""

    try:
        # Logging and seed
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.
        gym_env = create_env(flags)  # Create environment instance
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        #gym_env = wrappers.FullyObsWrapper(gym_env)

        # Create a stack of frames if the number of input frames is larger than 1
        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)

        # Observation_WrapperSetup turns the environment into a pytorch
        env = Observation_WrapperSetup(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed)
        # Dictionary with first observation
        env_output = env.initial()
        initial_frame = env_output['frame']
        agent_state = model.initial_state(batch_size=1)

        # Initialize first environment observation
        if flags.model != 'novelty_based':
            # Get the goal from the teacher
            generator_output = generator_model(env_output)
            goal = generator_output["goal"]
            agent_output, unused_state = model(env_output, agent_state, goal)
        else:
            agent_output, unused_state = model(env_output, agent_state)

        while True:
            # Remove and return an item from the queue associating to free indexes in the experience buffer
            index = free_queue.get()
            if index is None:
                break

            # Perform the new rollout
            for t in range(flags.unroll_length+1):
                # Update the buffer with the values of the results of the previous step
                for key in env_output:
                    buffers[key][index][t, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t, ...] = agent_output[key]
                if flags.model != 'novelty_based':
                    for key in generator_output:
                        buffers[key][index][t, ...] = generator_output[key]
                buffers["initial_frame"][index][t, ...] = initial_frame

                # Update the episodic buffer (used for the novelty-based intrinsic reward)
                episode_state_key = tuple(env_output['frame'].view(-1).tolist())
                if episode_state_key in episode_state_count_dict:
                    episode_state_count_dict[episode_state_key] += 1  # Update count if already there
                else:
                    episode_state_count_dict.update({episode_state_key: 1})
                buffers['episode_state_count'][index][t, ...] = \
                    torch.tensor(episode_state_count_dict.get(episode_state_key))
                # Reset the episode state counts when the episode is over
                if env_output['done'][0].item():
                    episode_state_count_dict = dict()

                timings.reset()

                if flags.model != 'novelty_based':
                    # True if the goal is reached when the associated tile is modified by the agent
                    if flags.modify:
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

                    # Generate new goal when reached intrinsic goal
                    if reached_condition:
                        if flags.restart_episode:  # If the episode should be restarted as a whole after reaching the goal
                            env_output = env.initial()
                        else:
                            env.episode_step = 0  # Reset the number of steps in the episode
                        initial_frame = env_output['frame']
                        # Inference, so no requirement for the gradient
                        with torch.no_grad():
                            # Generate new goal
                            generator_output = generator_model(env_output)  # Predict the new goal
                        goal = generator_output["goal"]

                if env_output['done'][0] == 1:  # Generate a New Goal when episode is finished
                    # Set the frame as the new initial_frame for the next iteration
                    initial_frame = env_output['frame']
                    if flags.model != 'novelty_based':
                        with torch.no_grad():
                            generator_output = generator_model(env_output)
                        goal = generator_output["goal"]

                # If agent has not lost the episode, predict action
                with torch.no_grad():
                    if flags.model != 'novelty_based':
                        agent_output, agent_state = model(env_output, agent_state, goal)
                    else:
                        agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")
                # Perform step in environment
                env_output = env.step(agent_output["action"])

                timings.time("step")
                timings.time("write")
            # Put the occupied index of the batch filled back into the full queue of processes
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


'''
The learn function implements the AMIGo loop and batch learning in order to optimize the policy network 
'''


def learn(
    actor_model, model, actor_generator_model, generator_model, random_target_network, predictor_network, batch,
        initial_agent_state, optimizer, generator_model_optimizer, predictor_optimizer, scheduler, generator_scheduler,
        flags, max_steps=100.0, lock=threading.Lock()):

    """Performs a learning (optimization) step for the policy, and for the generator whenever the generator batch is full."""
    with lock:
        # The novelty-based intrinsic reward like BeBold
        if flags.model == 'novelty_based':
            # Rnd intrinsic reward like BeBold
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
            # Novelty of states t and t+1
            rnd_novelty_tplus1 = torch.norm(predicted_embedding_tplus1.detach() - random_embedding_tplus1.detach(),
                                            dim=2, p=2)
            rnd_novelty_t = torch.norm(predicted_embedding_t.detach() - random_embedding_t.detach(), dim=2, p=2)
            # Episodic count mask
            mask_intrinsic_reward = batch['episode_state_count'][1:] == 1
            # Novelty-based intrinsic reward
            clamped_rnd_novelty = torch.clamp(rnd_novelty_tplus1 - rnd_novelty_t, min=0, max=None)
            intrinsic_rewards = 0.1 * clamped_rnd_novelty * mask_intrinsic_reward
            # Compute rnd loss
            rnd_loss = compute_forward_dynamics_loss(predicted_embedding_tplus1, random_embedding_tplus1.detach())

        else:
            # Traditional reward by AMIGo
            next_frame = batch['frame'][1:].float().to(device=flags.device)
            initial_frames = batch['initial_frame'][1:].float().to(device=flags.device)
            done_aux = batch['done'][1:].float().to(device=flags.device)  # Get where the experience was done

            # Get, for each training frame, whether the goal was reached or not
            reached_goal = reached_goal_func(next_frame, batch['goal'][1:].to(device=flags.device),
                                             modify = flags.modify, no_boundary_awareness=flags.no_boundary_awareness,
                                             initial_frames = initial_frames, done_aux = done_aux)
            reached = reached_goal.type(torch.bool)
            # The reward obtained by the agent intrinsically is awarded only when the agent reaches a goal
            intrinsic_rewards = flags.intrinsic_reward_coef * reached_goal
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
        
        # Compute total rewards
        total_rewards = rewards + intrinsic_rewards

        # Perform reward clipping
        clipped_rewards = torch.clamp(total_rewards, -1, 1)

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
        # Compute loss as the sum of the baseline loss, the policy
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
        if flags.model == 'novelty_based':
            total_loss = pg_loss + baseline_loss + entropy_loss + rnd_loss
        else:
            total_loss = pg_loss + baseline_loss + entropy_loss

        # Get the returns for the episodes by fetching timesteps flagged as "done"
        episode_returns = batch["episode_return"][batch["done"]]

        if torch.isnan(torch.mean(episode_returns)):
            aux_mean_episode = 0.0
        else:
            aux_mean_episode = torch.mean(episode_returns).item()

        # Statistics to be logged, different based on the chosen model
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": aux_mean_episode,
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_intrinsic_rewards":torch.mean(intrinsic_rewards).item()
        }
        if flags.model != 'novelty_based':
            stats.update({
                "gen_rewards": None,
                "gg_loss": None,
                "generator_baseline_loss": None,
                "generator_entropy_loss": None,
                "mean_episode_steps": None,
                "ex_reward": None,
                "generator_current_target": None,
                })

        # Perform gradient-based update
        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        # Set a maximum for the values of the parameters of the student
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        if flags.model == 'novelty_based':
            nn.utils.clip_grad_norm_(predictor_network.parameters(), 40.0)
            predictor_optimizer.step()
        # Share parameters of the learner with the actor model (the one performing rollouts)
        actor_model.load_state_dict(model.state_dict())

        # If the generator is present in the model
        if flags.model != 'novelty_based':
            global generator_batch
            global generator_batch_aux
            global generator_current_target
            global generator_count_increment
            global generator_count_decrement

            # Loading generator batch
            is_done = batch['done']==1
            # Reached is a variable of bools for all timesteps in all dimensions stating
            # whether the agent reached the goal or not at a certain time T in a batch B
            reached = reached_goal.type(torch.bool)
            keys = ['frame', 'episode_step', 'generator_logits', 'carried_obj', 'carried_col']
            if 'frame' in generator_batch.keys():
                for key in keys:
                    generator_batch[key] = torch.cat((generator_batch[key], batch[key][is_done].float().to(device=flags.device)), 0)
                    generator_batch[key] = torch.cat((generator_batch[key], batch[key][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][is_done].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'],torch.zeros(batch['goal'].shape)[is_done].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'],torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][is_done].to(device=flags.device)), 0)
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
            else:
                for key in keys:
                    if key != 'frame':
                        generator_batch[key] = (batch[key][is_done]).float().to(device=flags.device)
                        generator_batch[key] = torch.cat((generator_batch[key], batch[key][reached].float().to(device=flags.device)), 0)
                    generator_batch['ex_reward'] = (batch['reward'][is_done]).float().to(device=flags.device)
                    generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0)     
                    generator_batch['frame'] = (batch['initial_frame'][is_done]).float().to(device=flags.device)
                    generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0)
                    generator_batch['reached'] = (torch.zeros(batch['goal'].shape)[is_done]).float().to(device=flags.device)
                    generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)
                    generator_batch['goal'] = (batch['goal'][is_done]).to(device=flags.device)
                    generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)

            # Run Gradient step, keep batch residual in batch_aux
            if generator_batch['frame'].shape[0] >= flags.generator_batch_size:  # Run Gradient step, keep batch residual in batch_aux
                for key in generator_batch:
                    # Keep only a batch of pre-defined size in the generator_batch and place the rest in the auxiliary one
                    generator_batch_aux[key] = generator_batch[key][flags.generator_batch_size:]
                    generator_batch[key] = generator_batch[key][:flags.generator_batch_size].unsqueeze(0)

                generator_outputs = generator_model(generator_batch)
                # Get the generator value from the output of the generator model
                generator_bootstrap_value = generator_outputs["generator_baseline"][-1]

                def gen_reward(episode_step, reached, targ, adaptive = False):
                    """The function implementing the reward system for the agent"""
                    aux = flags.generator_reward_negative * torch.ones(episode_step.shape).to(device=flags.device)
                    if not adaptive:
                        aux += (episode_step >= targ).float() * reached
                    else:
                        # If adaptive, reward if the number of steps fit a window around t*
                        aux += torch.logical_and(torch.tensor(episode_step >= targ - flags.window),
                                                 torch.tensor(episode_step <= targ + flags.window)).float()
                    return aux

                # Calculate the generator rewards
                adaptive = False
                if flags.model == 'window_adaptive':
                    adaptive = True
                generator_rewards = torch.tensor(gen_reward(generator_batch['episode_step'], generator_batch['reached'],
                                                            targ=generator_current_target, adaptive = adaptive)).to(device=flags.device)

                # If the mean reward is higher than a pre-defined threshold, we increase the difficulty
                if torch.mean(generator_rewards).item() >= flags.generator_threshold:
                    if flags.model == 'default':
                        generator_count_increment += 1
                    else:
                        generator_count_decrement += 1
                else:
                    if flags.model == 'default':
                        generator_count_increment = 0
                    else:
                        generator_count_increment += 1

                # Increasing difficulty means increasing t*
                if flags.model == 'default':
                    if generator_count_increment >= flags.generator_counts and generator_current_target<=flags.generator_maximum:
                        generator_current_target += 1.0
                        generator_count_increment = 0
                else:
                    # Increasing mechanism of t* if the adaptive window method is chosen
                    if (generator_count_decrement >= flags.generator_counts or torch.mean(
                            generator_batch['ex_reward']) >= 0.8) and generator_current_target > 20:
                        generator_current_target -= 1.0
                        generator_count_decrement = 0
                        generator_count_increment = 0

                    elif generator_count_increment >= flags.generator_counts and generator_current_target < 270:
                        generator_current_target += 1.0
                        generator_count_decrement = 0
                        generator_count_increment = 0

                # Prepare for IMPALA
                # Clamp generator rewards
                generator_clipped_rewards = torch.clamp(generator_rewards, -1, 1)
                generator_clipped_rewards = 1.0 * (generator_batch['ex_reward'] > 0).float() + generator_clipped_rewards * \
                                            (generator_batch['ex_reward'] <= 0).float()
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

                generator_total_loss = gg_loss + generator_entropy_loss + generator_baseline_loss

                # Update the statistics to log
                intrinsic_rewards_gen = generator_batch['reached']*(1 - 0.9 * (generator_batch["episode_step"].float()/max_steps))
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

    # Create the RND-networks and the generator model if required
    if flags.model == 'novelty_based':
        random_target_network = RandomDistillationNetwork(env.observation_space.shape).to(device=flags.device)
        predictor_network = RandomDistillationNetwork(env.observation_space.shape).to(device=flags.device)
        generator_model = None
    else:
        random_target_network, predictor_network = None, None
        generator_model = TeacherNet(env.observation_space.shape, env.width, env.height,
                                     num_input_frames=flags.num_input_frames)

    # Initiaize student model
    no_generator = False
    if flags.model == 'novelty_based':
        no_generator = True

    model = StudentNet(env.observation_space.shape, env.action_space.n, goal_dim = flags.goal_dim,
                       no_generator = no_generator, state_embedding_dim=flags.state_embedding_dim,
                       num_input_frames=flags.num_input_frames,use_lstm=flags.use_lstm,
                       num_lstm_layers=flags.num_lstm_layers)

    # Define the size of the logits as the one of the board
    if flags.inner:
        logits_size = (env.width-2)*(env.height-2)
    else:  
        logits_size = env.width * env.height

    # Call create buffers function with well-defined parameters
    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags, env.width, env.height, logits_size)

    # All the processes of the multi-process run will share data from the buffer
    model.share_memory()
    if flags.model != 'novelty_based':
        generator_model.share_memory()

    # Add initial RNN state (not applied).
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # Deal with the multi-processing framework
    actor_processes = []
    # Create a multiprocessing fork object and the relative queues
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    episode_state_count_dict = dict()  # episodic counts
    # Generate different actors as data sharing processes and start them
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, generator_model, buffers, episode_state_count_dict,
                  initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    # Reassign the Net object to learner_model
    learner_model = StudentNet(env.observation_space.shape, env.action_space.n, goal_dim = flags.goal_dim,
                       no_generator = no_generator, state_embedding_dim=flags.state_embedding_dim,
                       num_input_frames=flags.num_input_frames,use_lstm=flags.use_lstm,
                       num_lstm_layers=flags.num_lstm_layers).to(device=flags.device)
    if flags.model != 'novelty_based':
        learner_generator_model = TeacherNet(env.observation_space.shape, env.width, env.height,
                                             num_input_frames=flags.num_input_frames).to(device=flags.device)
    else:
        learner_generator_model = None

    # Define optimizer variables for gradient propagation
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    
    if flags.model != 'novelty_based':
      generator_model_optimizer = torch.optim.RMSprop(
          learner_generator_model.parameters(),
          lr=flags.generator_learning_rate,
          momentum=flags.momentum,
          eps=flags.epsilon,
          alpha=flags.alpha)
      predictor_optimizer = []
    
    # Optimizer for the predictor network
    else:
      predictor_optimizer = torch.optim.RMSprop(
          predictor_network.parameters(),
          lr=flags.learning_rate,
          momentum=flags.momentum,
          eps=flags.epsilon,
          alpha=flags.alpha)
      generator_model_optimizer = []

    def lr_lambda(epoch):
        """Scheduling for alpha"""
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    # Adjust the scheduling of the lambda parameter
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if flags.model != 'novelty_based':
        generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_model_optimizer, lr_lambda)
    else:
      generator_scheduler = []

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "mean_intrinsic_rewards",
    ]
    if flags.model != 'novelty_based':
        stat_keys.extend([
            "gen_rewards",
            "gg_loss",
            "generator_entropy_loss",
            "generator_baseline_loss",
            "mean_episode_steps",
            "ex_reward",
            "generator_current_target"
            ])
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

            stats = learn(model, learner_model, generator_model, learner_generator_model, random_target_network,
                          predictor_network, batch, agent_state, optimizer, generator_model_optimizer, predictor_optimizer,
                          scheduler, generator_scheduler, flags, env.max_steps)

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
        """Regulate the logging of the results."""
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        # Save model weights
        dict_to_save = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            }
        if flags.model != 'novelty_based':
            dict_to_save.update({
                "generator_model_state_dict": generator_model.state_dict(),
                "generator_model_optimizer_state_dict": generator_model_optimizer.state_dict(),
                "generator_scheduler_state_dict": generator_scheduler.state_dict()
            })
        # Save model checkpoint to provided path 
        torch.save(dict_to_save, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        
        while frames < flags.total_frames:
            # Log once every 10 minutes from the previous checkpoint
            start_frames = frames
            start_time = timer()
            time.sleep(5) 
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

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

'''
TEST FUNCTION
'''


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
    if flags.model != 'novelty_based':
        teacher = TeacherNet(gym_env.observation_space.shape, gym_env.width, gym_env.height,
                        num_input_frames=flags.num_input_frames)
    student = StudentNet(gym_env.observation_space.shape, gym_env.action_space.n,
                             state_embedding_dim=flags.state_embedding_dim,
                             num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm,
                             num_lstm_layers=flags.num_lstm_layers)
    # Parametrize the policies
    state_dict = torch.load(flags.weight_path, map_location=torch.device(flags.device))
    student.load_state_dict(state_dict['model_state_dict'])
    if flags.model != 'novelty_based':
        teacher.load_state_dict(state_dict['generator_model_state_dict'])

    # Let teacher and student do a first step in the environment
    agent_state = student.initial_state(batch_size=1)
    if flags.model != 'novelty_based':
        generator_output = teacher(env_output)
        goal = generator_output["goal"]
        agent_output, unused_state = student(env_output, agent_state, goal)
    else:
        agent_output, unused_state = student(env_output, agent_state)

    # First step prediction
    while not done:
        # Check if the agent has reached the goal
        if flags.model != 'novelty_based':
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
                with torch.no_grad():
                    generator_output = teacher(env_output)
                goal = generator_output["goal"]

            with torch.no_grad():
                agent_output, agent_state = student(env_output, agent_state, goal)

        # Perform step
        else:
            with torch.no_grad():
                agent_output, agent_state = student(env_output, agent_state)

        gym_env.render()
        if flags.record_video:
            video_recorder.capture_frame()

        env_output = env.step(agent_output["action"])
        done = env_output['done']
    video_recorder.close()
    video_recorder.enabled = False
    return

'''
Initiate the training method
'''
def main(flags):
    """Call the train or test function"""
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
