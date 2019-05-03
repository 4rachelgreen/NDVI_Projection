# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:30:22 2019

@author: Rachel
"""
#%% Imports
    import os
    import numpy as np
    import pandas as pd


    import statsmodels.api as sm


    from scipy import stats
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates
    from sklearn import preprocessing


    %matplotlib qt5



#%%

os.chdir('C:/Users/grad/Documents/Courses/GEOG214A/Final_Project')

arsi = pd.read_csv('arsi.csv', parse_dates =['DT'])
bale = pd.read_csv('bale.csv')
borena = pd.read_csv('borena.csv')
guji = pd.read_csv('guji.csv')
westarsi = pd.read_csv('westarsi.csv')

DT = arsi['DT']
variables = arsi.drop('DT', axis = 1)
names = variables.columns

#standardized variables
#should i standardize using the StandardScaler, scale or MinMaxScaler, robustscalar, powertransformer
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(columns)
scaled_df = pd.DataFrame(scaled_df, columns = names)
final_df = scaled_df.join(DT)


myFmt = DateFormatter("%m/%y") 

months = mdates.MonthLocator()
fig, ax = plt.subplots(figsize = (10,8))
ax.plot(arsi.index, arsi.NDVI)
ax.plot(arsi.index, arsi.P)
sns.set_palette("cubehelix")
ax.set(xlabel="Date", ylabel="NDVI")
ax.set(title="Arsi")
ax.xaxis.set_minor_locator(months)
fig.autofmt_xdate()
#datemin = np.datetime64(arsi.index[0], 'Y')
#datemax = np.datetime64(arsi.index[-1], 'Y') + np.timedelta64(1, 'Y')
#ax.set_xlim(datemin, datemax)

#ax.xaxis.set_major_formatter(myFmt)
#%%

plt.style.use('seaborn-white')
my_dpi=96
plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
# multiple line plot
for column in final_df.drop('DT', axis=1):
   plt.plot(final_df['DT'], final_df[column], marker='', color='grey', linewidth=1, alpha=0.4)

# Now re do the interesting curve, but biger with distinct color
plt.plot(final_df['DT'], final_df['NDVI'], marker='', color='orange', linewidth=4, alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Environmental Variable")
 
#%%

# Initialize the figure
plt.style.use('seaborn-white')
 
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in final_df.drop('DT', axis=1):
    num+=1
 
    # Find the right spot on the plot
    plt.subplot(2,2, num)
 
    # Plot the lineplot
    plt.plot(final_df['DT'], final_df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)
 
    # Same limits for everybody!
    
    plt.ylim(-2,4)
 
    # Not ticks everywhere
  #  if num in range(3) :
  #      plt.tick_params(labelbottom='off')
  #  if num not in [1,4,7] :
  #      plt.tick_params(labelleft='off')
 
    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )
 
# general title
plt.suptitle("Arsi", fontsize=13, fontweight=0, color='black')
 
# Axis title
plt.text(0.5, 0.02, 'Time', ha='center', va='center')
plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')



