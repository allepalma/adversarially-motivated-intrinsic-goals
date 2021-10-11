import pandas as pd
import matplotlib.pyplot as plt

print('Reading data frame...')
df = pd.read_csv('logs.csv')

figure = plt.figure(figsize = (10,10))
plt.plot(df['frames'], df['mean_episode_return'], color = 'darkturquoise')
plt.xlabel('Frame', fontsize = 13)
plt.ylabel('Mean episode return', fontsize = 13)
plt.title('Mean episode return vs Frames', fontsize = 28)
plt.show()
