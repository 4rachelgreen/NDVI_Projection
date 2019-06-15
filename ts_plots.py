#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Mon May  6 17:28:34 2019'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''


"""

Name:           filename.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Rachel Green
ORGANIZATION:   University of California, Santa Barbara
Contact:        rachel.green@geog.ucsb.edu
Copyright:      (c) Rachel Green 2019


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

os.chdir('/Users/rgreen/Documents/Github/NDVI_Projection/')

oromia = pd.ExcelFile('oromia.xlsx')
arsi = pd.read_excel(oromia, 'arsi')
bale = pd.read_excel(oromia, 'bale')
borena = pd.read_excel(oromia, 'borena')
guji = pd.read_excel(oromia, 'guji')
westarsi = pd.read_excel(oromia, 'westarsi')

DT = arsi['DT']
variables = arsi.iloc[:,1:5]
names = variables.columns

#standardized variables
#should i standardize using the StandardScaler, scale or MinMaxScaler, robustscalar, powertransformer
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(variables)
scaled_df = pd.DataFrame(scaled_df, columns = names)
final_df = scaled_df.join(DT)

# =============================================================================
# f, axarr = plt.subplots(2, sharex=True)
# f.suptitle('Sharing X axis')
# axarr[0].plot(x, y)
# axarr[1].scatter(x, y)
# =============================================================================

fig, ax = plt.subplots(2)
ax[0].plot(final_df.DT, final_df.NDVI, 'forestgreen', label='NDVI', linewidth=2)
ax[0].plot(final_df.DT, final_df.ET, 'slateblue',  label='ET', alpha = 0.4)
ax[0].legend(loc = 'upper right')
ax[1].plot(final_df.DT[540:576], final_df.NDVI[540:576], 'forestgreen', label='NDVI', linewidth=2)
ax[1].plot(final_df.DT[540:576], final_df.ET[540:576], 'slateblue', label='ET', linewidth=2, alpha = 0.4)
ax[1].legend(loc = 'upper right')

#get yearly summaries


#%%

#soil moisture ts plot

DT2 = arsi['DT.1'][0:195]
variables2 = arsi.iloc[:,7:9][0:195]
names2 = variables2.columns

#standardized variables
#should i standardize using the StandardScaler, scale or MinMaxScaler, robustscalar, powertransformer
scaler = preprocessing.StandardScaler()
scaled_df2 = scaler.fit_transform(variables2)
scaled_df2 = pd.DataFrame(scaled_df2, columns = names2)
final_df2 = scaled_df2.join(DT2)


fig, ax = plt.subplots(2)
ax[0].plot(final_df2['DT.1'], final_df2['NDVI.1'], 'forestgreen', label='NDVI', linewidth=2)
ax[0].plot(final_df2['DT.1'], final_df2.SM, 'saddlebrown',  label='SM', alpha = 0.4)
ax[0].legend(loc = 'upper right')
ax[1].plot(final_df2['DT.1'][180:192], final_df2['NDVI.1'][180:192], 'forestgreen', label='NDVI', linewidth=2)
ax[1].plot(final_df2['DT.1'][180:192], final_df2.SM[180:192], 'saddlebrown', label='SM', linewidth=2, alpha = 0.4)
ax[1].legend(loc = 'upper right')





#%%

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



