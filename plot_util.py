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
env = 'OM'
# set directory to save plots
plots_dir = 'plot_results/'
# window for rolling mean
window = 5

# prepare directory for plots
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


frames = []
mean_episode_return = []
print('Collecting results...')
for filename in os.listdir(results_dir):
    if filename.startswith(env):
        print(os.path.join(results_dir, filename))
        path = os.path.join(results_dir, filename)
        data = pd.read_csv(os.path.join(path, 'logs.csv'))
        frames.append(data['frames'])
        mean_episode_return.append((data['mean_episode_return']))

# computing mean
mean_episode_return = np.mean(mean_episode_return, axis=0)
# TODO: not sure how precise that would be if different logging step was used in files
mean_frames = np.mean(frames, axis=0)

print('Plotting the results...')
plt.figure(figsize=(12, 8))
rolling_mean = pd.Series(mean_episode_return).rolling(window).mean()
std = pd.Series(mean_episode_return).rolling(window).std()
plt.plot(mean_frames, rolling_mean, color='darkturquoise')
plt.fill_between(mean_frames,rolling_mean-std, rolling_mean+std, color='paleturquoise', alpha=0.4)

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