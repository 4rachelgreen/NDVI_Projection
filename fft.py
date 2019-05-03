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
<<<<<<< HEAD

#Nyquist Freqency for low pass filter? 


=======
>>>>>>> 57a6fac9488171b799fce6713ea5ba37972cbc35
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

    

fs = 100 # sample rate 
f = 2 # the frequency of the signal

x = np.arange(fs) # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*np.pi*f * (x/fs)) 


# showing the exact location of the smaples

plt.plot(x,y)
    
sin_fft = sp.fftpack.fft(y)

sin_psd = np.abs(sin_fft) ** 2 #power spectral density

fftfreq = sp.fftpack.fftfreq(len(sin_psd), 1. / 100) # data

i = fftfreq > 0 #only want positive frequencies

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(sin_psd[i]))
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency')
ax.set_ylabel('PSD (dB)') #logarithmic y scale, decibels

#%%


df = pd.read_csv('/Users/rgreen/Documents/Github/NDVI_Projection/arsi.csv')
df.dropna(how="all", inplace=True)

df['DT'] = pd.to_datetime(df['DT'])
date = df.DT

#date = df.index.to_datetime()
df.set_index('DT', inplace=True)

#df2 = df.set_index('Date')

<<<<<<< HEAD
NDVI = df.NDVI

NDVI = df.iloc[:,0]


fig1, ax = plt.subplots(1, 1, figsize=(6, 3))
NDVI.plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('NDVI')

#%%
NDVI_fft = sp.fftpack.fft(NDVI)

NDVI_psd = np.abs(NDVI_fft) ** 2 #power spectral density

<<<<<<< HEAD
fftfreq = sp.fftpack.fftfreq(len(NDVI_psd), 1. / 36) # data

i = fftfreq > 0 #only want positive frequencies


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], NDVI_psd[i])
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)') 


#%%
#cut out frequencies higher than f=1
NDVI_fft_bis = NDVI_fft.copy()
NDVI_fft_bis[np.abs(fftfreq) > 1] = 0
#Do I need to do a more complicated low pass filter? 

#%%
#recover signal that mainly contains fundamental frequency
#convert back to time domain
NDVI_real = np.real(sp.fftpack.ifft(NDVI_fft_bis))
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
NDVI.plot(ax=ax)
ax.plot(date, NDVI_real, '-', label ='iFFT')

ax.set_xlabel('Date')
ax.set_ylabel('NDVI')
ax.legend(loc=1)

#%%


LST = df.LST

fig1, ax = plt.subplots(1, 1, figsize=(6, 3))
LST.plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('LST')

LST_fft = sp.fftpack.fft(LST)

LST_psd = np.abs(LST_fft) ** 2 #power spectral density

fftfreq = sp.fftpack.fftfreq(len(LST_psd), 1. / 36) # data frequency

i = fftfreq > 0 #only want positive frequencies


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], LST_psd[i])
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)') 

#cut out frequencies higher than f=1
LST_fft_bis = LST_fft.copy()
LST_fft_bis[np.abs(fftfreq) > 1] = 0
#Do I need to do a more complicated low pass filter? 

#recover signal that mainly contains fundamental frequency
#convert back to time domain
LST_real = np.real(sp.fftpack.ifft(LST_fft_bis))
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
LST.plot(ax=ax)
ax.plot(date, LST_real, '-', label ='iFFT')

ax.set_xlabel('Date')
ax.set_ylabel('LST')
ax.legend(loc=1)


#%%

#Sub annual FFT

LST_s = df.LST.iloc[0:36]

fig1, ax = plt.subplots(1, 1, figsize=(6, 3))
LST_s.plot(ax=ax)
ax.set_xlabel('Date')
ax.set_ylabel('LST 2003')

LST_fft = sp.fftpack.fft(LST_s)

LST_psd = np.abs(LST_fft) ** 2 #power spectral density

fftfreq = sp.fftpack.fftfreq(len(LST_psd), 1. / 36) # data frequency

i = fftfreq > 0 #only want positive frequencies


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], LST_psd[i])
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)') 

#cut out frequencies higher than f=1
LST_fft_bis = LST_fft.copy()
LST_fft_bis[np.abs(fftfreq) > 1] = 0
#Do I need to do a more complicated low pass filter? 

#recover signal that mainly contains fundamental frequency
#convert back to time domain
LST_real = np.real(sp.fftpack.ifft(LST_fft_bis))
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
LST.plot(ax=ax)
ax.plot(date, LST_real, '-', label ='iFFT')

ax.set_xlabel('Date')
ax.set_ylabel('LST')
ax.legend(loc=1)









#%%
#Examples
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

#%%%

N = 256
t = np.arange(N)

m = 4
nu = float(m)/N
f = np.sin(2*np.pi*nu*t)
ft = np.fft.fft(f)
freq = np.fft.fftfreq(N)
plt.plot(freq, ft.real**2 + ft.imag**2)
plt.show()

#%%
"""
=============================================
Plotting and manipulating FFTs for filtering
=============================================

Plot the power of the FFT of a signal and inverse FFT back to reconstruct
a signal.

This example demonstrate :func:`scipy.fftpack.fft`,
:func:`scipy.fftpack.fftfreq` and :func:`scipy.fftpack.ifft`. It
implements a basic filter that is very suboptimal, and should not be
used.

"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

############################################################
# Generate the signal
############################################################

# Seed the random number generator
np.random.seed(1234)

time_step = 0.02
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(2 * np.pi / period * time_vec)
       + 0.5 * np.random.randn(time_vec.size))

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')

############################################################
# Compute and plot the power
############################################################

# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection

############################################################
# Remove all the high frequencies
############################################################
#
# We now remove all the high frequencies and transform back from
# frequencies to signal.

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

############################################################
#
# **Note** This is actually a bad way of creating a filter: such brutal
# cut-off in frequency space does not control distorsion on the signal.
#
# Filters should be created using the scipy filter design code
plt.show()

#%%

"""
==========================
Crude periodicity finding
==========================

"""

############################################################
# Load the data
############################################################

import numpy as np
data = pd.read_csv('/Users/rgreen/Documents/Github/NDVI_Projection/arsi_sm.csv')
date = data.DT
env = data[['NDVI','SM']]
NDVI = data.NDVI
data['time'] = range(1,len(data)+1) #convert dates to just counts (from 1/1/03-3/1/19)
time = data.time

############################################################
# Plot the data
############################################################

import matplotlib.pyplot as plt
plt.figure()
plt.plot(time, NDVI)
plt.xlabel('Date')
plt.ylabel('Env variable')
plt.legend(['NDVI'], loc=1)

############################################################
# Plot its periods
############################################################
from scipy import fftpack

ft_env = fftpack.fft(NDVI, axis=0)
frequencies = fftpack.fftfreq(env.shape[0], time[1] - time[0])
periods = 1 / frequencies

plt.figure()
plt.plot(periods, abs(ft_env), 'o')
plt.xlim(0, 22)
plt.xlabel('Period')
plt.ylabel('Power')
plt.legend(['NDVI'], loc=1)

plt.show()

############################################################
#for NDVI period seems to be 12.5 dekads ~4.16 months

#%%

#Testing out on the RS data 


import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

############################################################
# Generate the signal
############################################################

arsi = pd.read_csv('/Users/rgreen/Documents/Github/NDVI_Projection/arsi.csv')

time_step = 0.02
period = 5.

time_vec = arsi.DT
sig = arsi.NDVI

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')

############################################################
# Compute and plot the power
############################################################

# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])

# scipy.signal.find_peaks_cwt can also be used for more advanced
# peak detection

############################################################
# Remove all the high frequencies
############################################################
#
# We now remove all the high frequencies and transform back from
# frequencies to signal.

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')

############################################################
#
# **Note** This is actually a bad way of creating a filter: such brutal
# cut-off in frequency space does not control distorsion on the signal.
#
# Filters should be created using the scipy filter design code
plt.show()



=======
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
>>>>>>> 57a6fac9488171b799fce6713ea5ba37972cbc35

#%%
if __name__ == "__main__":
    main()



#%
