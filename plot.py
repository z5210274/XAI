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

headers1 = ['Episode','Reward','Time']
headers2 = ['Episode', 'Running_Reward', 'Frame_Count', 'Time']
df1 = pd.read_csv(filename,names=headers1)
df2 = pd.read_csv(filename2,names=headers2)

df1 = df1.iloc[1:, :]
df2 = df2.iloc[1:, :]

x = df1['Episode'].astype(int)
y = df1['Reward'].astype(float)

y_average = y.rolling(window=50).mean()

plt.plot(x,y)
plt.plot(x,y_average, color='r')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title("Episode vs Reward")
plt.show()

x = df2['Episode'].astype(int)
y = df2['Running_Reward'].astype(float)

plt.plot(x,y)
plt.xlabel('Episode')
plt.ylabel('Running_Reward')
plt.title("Target Episode vs Running_Reward")
plt.show()