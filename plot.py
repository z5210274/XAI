import matplotlib.pyplot as plt
import os.path
import numpy as np
import pandas as pd
import csv
from datetime import datetime

filename = './episode.csv'
filename2 = './target.csv'

check_file = os.path.isfile(filename)
check_file2 = os.path.isfile(filename2)
print("episode.csv exists: " + str(check_file))
print("target.csv exists: " + str(check_file2))

headers1 = ['Episode','Reward','Time', 'Hits', 'Avg_pos', 'Avg_theta', 'Shots']
headers2 = ['Episode', 'Running_Reward', 'Frame_Count', 'Time']
df1 = pd.read_csv(filename,names=headers1)
df2 = pd.read_csv(filename2,names=headers2)

df1 = df1.iloc[1:, :]
df2 = df2.iloc[1:, :]

x = df1['Episode'].astype(int)
y = df1['Reward'].astype(float)
y2 = df1['Hits'].astype(int)
y3 = df1['Avg_pos'].astype(float)
y4 = df1['Avg_theta'].astype(float)
y5 = df1['Shots'].astype(float)

y_average = y.rolling(window=50).mean()

plt.plot(x,y)
plt.plot(x,y_average, color='r', label = '50 Episodes Avg Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title("Episode vs Reward")
plt.legend(loc="upper left")
plt.show()

plt.plot(x,y2, color = 'b', label = 'Shots Hit')
plt.plot(x,y3, color = 'r', label = "Avg Positional Diff")
plt.plot(x,y4, color = 'g', label = 'Avg Theta Diff')
plt.plot(x,y5, color = 'm', label = 'Shots Taken')
plt.xlabel('Episode')
plt.ylabel('Goals Tracker')
plt.title("Shots hit per Episode")
plt.legend(loc="upper left")
plt.show()

x = df2['Episode'].astype(int)
y = df2['Running_Reward'].astype(float)

plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Running_Reward')
plt.title("Target Episode vs Running_Reward")
plt.show()