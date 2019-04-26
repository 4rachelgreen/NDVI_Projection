#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Rachel Green'
__contact__ = 'rachel.green@geog.ucsb.edu'
__copyright__ = '(c) Rachel Green 2019'

__license__ = 'MIT'
__date__ = 'Tue Apr 23 13:30:19 2019'
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

#based on iPython tutorial: https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/
#%% IMPORTS

import datetime
import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt5

#%% FUNCTIONS



#%% MAIN
def main():
#%%

df = pd.read_csv('/Users/rgreen/Documents/Github/NDVI_Projection/arsi.csv')
df.dropna(how="all", inplace=True)

df['DT'] = pd.to_datetime(df['DT'])
date = df.DT

#date = df.index.to_datetime()
df.set_index('DT', inplace=True)

#df2 = df.set_index('Date')

NDVI = df.iloc[:,0]

fig1, ax = plt.subplots(1, 1, figsize=(6, 3))
NDVI.plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('NDVI')

#%%
NDVI_fft = sp.fftpack.fft(NDVI)

NDVI_psd = np.abs(NDVI_fft) ** 2 #power spectral density

fftfreq = sp.fftpack.fftfreq(len(NDVI_psd), 1. / 36) #monthly data

i = fftfreq > 0 #only want positive frequencies

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(NDVI_psd[i]))
#ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/dekad)')
ax.set_ylabel('PSD (dB)') #logarithmic y scale, decibels

#do i need to have the year starting at January? 
#how to convert back to time

#looks like peak frequency is at 2 - what does this mean

#%%

NDVI_fft_bis = NDVI_fft.copy()
NDVI_fft_bis[np.abs(fftfreq) > 1.1] = 0

#%%

NDVI_real = np.real(sp.fftpack.ifft(NDVI_fft_bis))
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
NDVI.plot(ax=ax)
ax.plot_date(date, NDVI_real, '-')
ax.set_xlim(datetime.DT(2003, 1, 10),
            datetime.DT(2019, 3, 31))
#ax.set_ylim(-10, 40)
ax.set_xlabel('Date')
ax.set_ylabel('NDVI')

#%%
if __name__ == "__main__":
    main()



#%
