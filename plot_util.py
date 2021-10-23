import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

############## SETTINGS ################
# set directory to read
results_dir = 'results/'
# dictionary of environments
env_dict = {
    'KC': 'KeyCorridorS3R3',
    'OM': 'ObstructedMaze-1Dl'
}
# set environment to plot, for now KC/OM
env = 'KC'
# set directory to save plots
plots_dir = 'plot_results/'
# window for rolling mean
window = 5

# Whether to plot the generator_reward, intrinsic reward and t* as well
plot_extras = True

# prepare directory for plots
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


frames = []
mean_episode_return = []
current_target = []
generator_rewards = []
mean_intrinsic_reward = []

print('Collecting results...')
for filename in os.listdir(results_dir):
    if filename.startswith(env):
        print(os.path.join(results_dir, filename))
        path = os.path.join(results_dir, filename)
        data = pd.read_csv(os.path.join(path, 'logs.csv'))
        # Appends
        frames.append(data['frames'])
        mean_episode_return.append((data['mean_episode_return']))
        current_target.append((data['generator_current_target']))
        generator_rewards.append((data['gen_rewards']))
        mean_intrinsic_reward.append((data['mean_intrinsic_rewards']))


# computing mean
avg_mean_episode_return = np.mean(mean_episode_return, axis=0)
# compute std for confidence interval as std/sqrt(n) where n is the number of repetitions
std_episode_return = np.std(mean_episode_return, axis = 0)/np.sqrt(len(mean_episode_return))

print('Plotting the results...')
plt.figure(figsize=(12, 8))
rolling_mean = pd.Series(avg_mean_episode_return).rolling(window).mean()
rolling_ci = pd.Series(std_episode_return).rolling(window).mean()
plt.plot(frames[0], rolling_mean, color='darkturquoise')
plt.fill_between(frames[0],rolling_mean-rolling_ci, rolling_mean+rolling_ci, color='paleturquoise', alpha=0.4)

plt.title('Mean episode return vs Frames', fontsize = 28)
plt.xlabel('Frame', fontsize = 13)
plt.ylabel('Mean episode return', fontsize = 13)
# name figure according to environment and save to plots_dir
plt.savefig('{}{}_window{}.png'.format(plots_dir, env_dict[env], window))
# plt.show()
plt.clf()
print('Saved results to {}{}_window{}.png'.format(plots_dir, env_dict[env], window))

# print('Reading data frame...')
# df = pd.read_csv('logs.csv')
#
# figure = plt.figure(figsize = (10,10))
# plt.plot(df['frames'], df['mean_episode_return'], color='darkturquoise')
# plt.xlabel('Frame', fontsize = 13)
# plt.ylabel('Mean episode return', fontsize = 13)
# plt.title('Mean episode return vs Frames', fontsize = 28)
# plt.show()


# Plot as well the mean intrinsic reward, the generator reward and t*
if plot_extras:
    current_target = current_target[1].to_numpy()
    generator_rewards = generator_rewards[1].to_numpy()
    mean_intrinsic_reward = mean_episode_return[1].to_numpy()

    # Remove nans
    frames_not_nan = frames[0][~np.isnan(current_target)]
    current_target_not_nan = current_target[~np.isnan(current_target)]
    generator_rewards_not_nan = generator_rewards[~np.isnan(current_target)]
    mean_intrinsic_reward_not_nan = mean_intrinsic_reward[~np.isnan(current_target)]

    # Plot t* with respect to current number of frames
    plt.figure(figsize=(12, 8))
    plt.plot(frames_not_nan, current_target_not_nan, color='goldenrod')
    plt.title('t* vs Frames', fontsize=28)
    plt.xlabel('Frame', fontsize=18)
    plt.ylabel('t*', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('{}{}_window{}_t_star.png'.format(plots_dir, env_dict[env], window))
    plt.clf()

    #Plot generator rewards
    rolling_mean_generator_rewards = pd.Series(generator_rewards_not_nan).rolling(window).mean()
    plt.plot(frames_not_nan, rolling_mean_generator_rewards, color='orchid')
    plt.title('Generator reward vs Frames', fontsize=28)
    plt.xlabel('Frame', fontsize=18)
    plt.ylabel('Generator reward', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # name figure according to environment and save to plots_dir
    plt.savefig('{}{}_window{}_generator_reward.png'.format(plots_dir, env_dict[env], window))
    # plt.show()
    plt.clf()

    #Plot generator rewards
    rolling_mean_intrinsic_reward = pd.Series(mean_intrinsic_reward_not_nan).rolling(window).mean()
    plt.plot(frames_not_nan, rolling_mean_intrinsic_reward, color='seagreen')
    plt.title('Mean intrinsic reward vs Frames', fontsize=28)
    plt.xlabel('Frame', fontsize=18)
    plt.ylabel('Mean intrinsic reward', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # name figure according to environment and save to plots_dir
    plt.savefig('{}{}_window{}_mean_intrinsic_reward.png'.format(plots_dir, env_dict[env], window))
    # plt.show()
    plt.clf()
