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
#%%

#https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

#%% IMPORTS
from pandas import DataFrame
from pandas import concat


#%% FUNCTIONS

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#%% MAIN
def main():
 #%%


df = DataFrame()
df['t'] = [x for x in range(10)]
df['t-1'] = df['t'].shift(1) #shift down, past observations
#df['t+1'] = df['t'].shift(-1) #shift up, future time steps
print(df)

#%%

values = [x for x in range(10)]
data = series_to_supervised(values)
print(data)

#%%
#repeat this example with an arbitrary number length input sequence, such as 3
values = [x for x in range(10)]
data = series_to_supervised(values, 3)
print(data)


#%%

#Multi-Step or Sequence Forecasting
#Ex. frame a forecast problem with an input sequence of 2 past observations to forecast 2 future observations

values = [x for x in range(10)]
data = series_to_supervised(values, 2, 2)
print(data)

#%%

#Multivariate forecasting

raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values)
print(data)

#example of a reframing with 1 time step as input and 2 time steps as forecast sequence
raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(values, 1, 2)
print(data)

#%%
if __name__ == "__main__":
    main()



#%
